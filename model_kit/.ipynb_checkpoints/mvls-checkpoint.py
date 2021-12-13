import numpy as np
import copy
import scipy

from astropy.io import fits
from astropy import units as u
from datetime import datetime
import time

from skimage.draw import draw

from . import datafiles as dfx
from . import psd_functions as psd

class mvls:
    
    def __init__(self, surf_name):
        self.surf_name = surf_name
        
    def load_data_mask(self, data, mask, dx):
        # create the tn vector space
        self.dx = dx
        side = np.shape(mask)[0]
        self.side_data = side
        cen = int(side/2)
        if side % 2 == 0: # if even
            yy, xx = np.mgrid[-cen:cen, -cen:cen]
        else: #if odd, need to add 1 more that int removes
            yy, xx = np.mgrid[-cen:cen+1, -cen:cen+1]
        tnx = (xx * dx)
        tny = yy * dx
        self.tnx = tnx
        self.tny = tny
        
        # load the data and apply window to it
        surf_win = data * psd.han2d((side, side))
        
        # filter the data
        # Using mask_filter automatically vectorizes the data
        mask_filter = np.where(mask==1) # outputs vector location
        self.tn = np.vstack((tnx[mask_filter].value, tny[mask_filter].value))*dx.unit
        self.ytn = surf_win[mask_filter]
        self.ytn_var = np.var(self.ytn)
        
    def set_spatial_freq(self, k_side, dk=None):
        self.k_side = k_side # the main code sets k_side = data_side
        if dk==None:
            self.dk = 1/(self.side_data*self.dx) # always correct
        else:
            self.dk = dk
        k_cen = int(k_side/2)
        if k_side % 2 == 0:
            ky, kx = np.mgrid[-k_cen:k_cen, -k_cen:k_cen]
        else:
            ky, kx = np.mgrid[-k_cen:k_cen+1, -k_cen:k_cen+1]
        wkx = kx * self.dk
        wky = ky * self.dk
        self.wkx = np.reshape(wkx, k_side**2)
        self.wky = np.reshape(wky, k_side**2)
        
    def calc_mvls(self, print_update=False):
        k_tot = np.shape(self.wkx)[0]
        
        # localize without the units
        wkx = self.wkx.value
        wky = self.wky.value
        tn = self.tn.value
        ytn = self.ytn.value
        
        # initialize variables
        tau = np.zeros((k_tot))
        ak = np.zeros((k_tot))
        bk = np.zeros((k_tot))
        
        if print_update == True:
            print('Scargling in progress, startimg time =', 
                  datetime.now().strftime("%H:%M:%S")) 
        
        for nk in range(0, k_tot):
            # calculate dot product
            wkvec = ([wkx[nk], wky[nk]])
            wdt = np.dot(wkvec, tn) * 2 * np.pi #2pi required to set as radians
            
            # calculate tau from that
            tau_num = np.sum(np.cos(2*wdt))
            tau_denom = np.sum(np.sin(2*wdt))
            tau_val = 0.5 * np.arctan2(tau_num, tau_denom) 
            tau[nk] = tau_val
            
            # now calculate inner product for ak and bk
            inner_calc = wdt - tau_val
            akcos = np.cos(inner_calc)
            bksin = np.sin(inner_calc)
            
            # solve for ak
            ak_num = np.sum(ytn*akcos)
            ak_denom = np.sum(akcos**2)
            ak[nk] = ak_num/ak_denom
            
            # solve for bk
            bk_num = np.sum(ytn*bksin)
            bk_denom = np.sum(bksin**2)
            bk[nk] = bk_num/bk_denom
            
            if print_update == True:
                if nk % 10000 == 0:
                    print('{0} of {1} complete ({2:.2f}%), time ='.format(nk, 
                                                                          k_tot, 
                                                                          nk*100/k_tot)
                          , datetime.now().strftime("%H:%M:%S"))
        
        # save the variables into objects
        self.tau = tau
        self.ak = ak * self.ytn.unit
        self.bk = bk * self.ytn.unit
        
        # calculate the PSD
        psd = ((ak**2) + (bk**2)) / (self.dk.value**2)
        self.psd = np.reshape(psd, (self.k_side, self.k_side)) * (self.ytn.unit/self.dk.unit)**2
        
        if print_update==True:
            print('Scargling completed')
        
    def calc_psd(self):
        psd = ((self.ak.value**2) + (self.bk.value**2)) / (self.dk.value**2)
        self.psd = np.reshape(psd, (self.k_side, self.k_side)) * (self.ytn.unit/self.dk.unit)**2
        
    def calc_tau(self, print_update=False):
        k_tot = np.shape(self.wkx)[0]
        tau = np.zeros((k_tot))
        if print_update == True:
            print('Tau calculation progress')
        for nk in range(0, k_tot):
            wkvec = ([self.wkx[nk],self.wky[nk]])
            wdt = np.dot(wkvec, self.tn) * 2 * pi
            tau_num = np.sum(np.cos(2*wdt))
            tau_denom = np.sum(np.sin(2*wdt))
            tau[nk] = 0.5 * np.arctan2(tau_num,tau_denom)
            
            if print_update == True:
                if nk % 10000 == 0:
                    print('{0} of {1} complete ({2:.2f}%)'.format(nk, k_tot, nk*100/k_tot))
        self.tau = tau
    
    def calc_akbk(self, print_update=False):
        k_tot = np.shape(self.wkx)[0]
        ak = np.zeros((k_tot))
        bk = np.zeros((k_tot))
        if print_update == True:
            print('Ak Bk calculation progress')
        for nk in range(0, k_tot):
            wkvec = ([self.wkx[nk],self.wky[nk]])
            wdt = np.dot(wkvec, self.tn) * 2 * pi
            inner_calc = wdt - self.tau[nk]
            akcos = np.cos(inner_calc)
            bksin = np.sin(inner_calc)
            
            ak_num = np.sum(self.ytn*akcos)
            ak_denom = np.sum(akcos**2)
            ak[nk] = ak_num/ak_denom
            
            bk_num = np.sum(self.ytn*bksin)
            bk_denom = np.sum(bksin**2)
            bk[nk] = bk_num/bk_denom
            
            if print_update == True:
                if nk % 10000 == 0:
                    print('{0} of {1} complete ({2:.2f}%)'.format(nk, k_tot, nk*100/k_tot))
                          
        self.ak = ak
        self.bk = bk
    
        
                
        
        
