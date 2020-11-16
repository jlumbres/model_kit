'''
psd_functions.py
This modular function file has the following goals:
* [surfPSD] class that calculates PSD stuff for single surface
* [MODELING] Support functions for modeling a PSD
* [INTERPOLATION] The function in here needs to be checked.
* [SUPPORT] Dumping ground for functions running specific tasks in bigger functions
'''

import numpy as np
import copy
from scipy import interpolate
from scipy.optimize import least_squares
from datetime import datetime
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt


# for calling in data
from astropy.io import fits
from astropy import units as u

# drawing for the apertures
from skimage.draw import draw

# add in datafiles modular code
from . import datafiles as dfx

##########################################
# surfPSD Class definition

class surfPSD:
    
    def __init__(self, surf_name, **kwargs):
        self.surf_name = surf_name
        
    def open_surf(self, fileloc, surf_units):
        # to use if opening the data from a FITS file
        # assumes the data is efficiently filled (no zero col/row)
        self.data = (fits.open(fileloc)[0].data*surf_units) # units from Zgyo analysis
        hdr = fits.open(fileloc)[0].header
        self.wavelen = hdr['WAVELEN'] * u.m
        self.latres = hdr['LATRES'] * u.m / u.pix
    
    def load_surf(self, data, wavelen, latres):
        # to use if data has already been loaded into environment
        if hasattr(data, 'unit'): # set to data if there are units
            self.data = data
        else: # exit if there are no units
            raise Exception('Data needs units')
        if hasattr(wavelen, 'unit'):
            self.wavelen = wavelen
        else:
            raise Exception('Wavelength needs units')
        if hasattr(latres, 'unit'):
            self.latres = latres
        else:
            raise Exception('Lateral Resolution needs units')
        
    def open_mask(self, fileloc):
        mask = fits.open(fileloc)[0].data
        self.load_mask(mask)
    
    def load_mask(self, mask):
        if mask.shape != self.data.shape:
            raise Exception('Mask and data are not compatiable (shape)')
        else:
            self.mask = mask.astype(bool)
            self.npix_diam = int(np.sum(mask[int(mask.shape[0]/2)]))
            self.diam_ca = (self.npix_diam * u.pix * self.latres)#.to(u.mm)
            
    def open_psd(self, psd_fileloc, psd_type, var_unit = u.nm):
        psd_fits = fits.open(psd_fileloc)[0]
        hdr = psd_fits.header
        self.oversamp = hdr['oversamp']
        self.diam_ca = hdr['diam_ca']*u.mm
        self.npix_diam = int(hdr['diam_pix'])
        self.var = hdr['var_tot']*(var_unit**2)
        self.calc_psd_parameters()
        
        # set the PSD variables
        if psd_type == 'norm':
            self.psd_norm = psd_fits.data * (self.diam_ca.unit**2)
            self.psd_cal = self.psd_norm * self.var
        elif psd_type == 'cal':
            self.psd_cal = psd_fits.data * (self.diam_ca.unit**2) * (var_unit**2)
            
    def load_psd(self, psd_data, psd_type, var=None):
        if hasattr(psd_data, 'unit'):
            if psd_type=='norm':
                self.psd_norm=psd_data
                self.psd_cal = self.psd_norm * var 
            elif psd_type=='cal':
                self.psd_cal=psd_data
            elif psd_type=='raw':
                self.psd_raw=psd_raw
            else:
                raise Exception('Type does not match cal, norm, raw; type sent: {0}'.format(psd_type))
        else:
            raise Exception('PSD needs units')
        
    def load_psd_parameters(self, diam_ca, npix_diam, wavelen,delta_k=None, oversamp=None):
        self.diam_ca = diam_ca
        self.npix_diam = npix_diam
        self.wavelen = wavelen
        if oversamp is not None:
            self.oversamp = oversamp
        else:
            self.oversamp = np.shape(self.psd_cal.value)[0]
        self.calc_psd_parameters(delta_k=delta_k) # calculate other necessary parameters
    
    def calc_psd_parameters(self, delta_k=None):
        self.k_min = 1/self.diam_ca
        self.k_max = 1/(2*self.diam_ca / self.npix_diam)
        if delta_k is not None:
            self.delta_k = delta_k
        else:
            self.delta_k = 1/(self.oversamp*self.diam_ca/self.npix_diam)
        
    def calc_psd(self, oversamp, kmid_ll = 0.1/u.mm, khigh_ll=1/u.mm, var_unit = u.nm):
        self.oversamp = oversamp
        if var_unit != self.data.unit: # data units must match variance units.
            self.data = self.data.to(var_unit)
        # note - data MUST be even, square, and efficiently filled.
        optic = self.data.value

        # calculate the mean and variance of the active region of data
        ap_active = optic * self.mask
        ap_active_coords = optic[self.mask==True]
        ap_avg = np.mean(ap_active_coords)
        self.var = (np.var(ap_active_coords)*(self.data.unit**2))#.to(var_unit**2)

        # Subtract the mean from the data
        ap_submean = (ap_active - ap_avg) * self.mask

        # build the Hann 2D window and apply to surface
        hannWin = han2d((self.npix_diam, self.npix_diam)) * ap_submean 
        
        # before zero padding to FT, check that all the data is good inside the mask.
        # Otherwise, interpolate.
        n_badpix = np.where(np.isnan(hannWin)==True)[0].shape[0]
        if n_badpix > 0:
            print('Bad data present: {0} nan pixels. Interpolating to fix.'.format(n_badpix))
            ap_clear = copy.copy(self.mask)
            ap_clear[np.where(np.isnan(hannWin)==True)] = 1 # cover up nan holes
            ap_coords = np.where(ap_clear==1)
            hannWin = dfx.fill_surface(hannWin, self.mask, ap_clear, ap_coords)
        
        # zero pad to oversample then start taking FT's
        pad_side = (self.oversamp - self.mask.shape[0])/2
        if pad_side%1 != 0: # fractional pixel present for padding
            pad_side_a = int(np.floor(pad_side))
            pad_side_b = int(self.oversamp - self.mask.shape[0] - pad_side_a)
            optic_ovs = np.pad(hannWin, ((pad_side_a, pad_side_b), (pad_side_a, pad_side_b)), 
                               'constant', constant_values=0)
        else: # no fractional pixel present for padding
            pad_side = int(pad_side)
            optic_ovs = np.pad(hannWin, ((pad_side, pad_side), (pad_side, pad_side)), 
                               'constant', constant_values=0)
        assert (np.shape(optic_ovs)[0] == self.oversamp), "Padding does not match desired oversample size"
        
        FT_wf = np.fft.fftshift(np.fft.fft2(optic_ovs))*self.data.unit # FT is unitless
        self.psd_raw = np.real(FT_wf*np.conjugate(FT_wf))#/(self.data.unit**2)
        # psd_raw should be in units of [data]**2
        
        # The raw power is uncalibrated, need to normalize then multiply by variance.
        self.calc_psd_parameters() # calculate some basic parameters needed for PSDs and everything
        self.psd_norm = self.psd_raw / (np.sum(self.psd_raw)*(self.delta_k**2))
        # psd_norm units should be in units of [delta_k]**2
        self.psd_cal = self.psd_norm * self.var
        # psd_cal should be in units of [delta_k]**2 [data]**2
        
        # Calculate the RMS based on the k-parameter limits
        self.calc_rms_set(kmid_ll=kmid_ll, khigh_ll=khigh_ll, pwr_opt=self.psd_cal)
        # rms should be in units of [data]
        
    def check_normpsd(self, psd_norm):
        var_verify = np.sum(psd_norm) * (self.delta_k**2) # unitless and 1
        psd_verify = np.allclose(1, var_verify)
        if psd_verify==True:
            print('PSD normalized: var={0:.3f}'.format(var_verify))
        else:
            print('PSD not normalized: var={0:.3f}. What happened?'.format(var_verify))
    
    def mask_psd(self, center, radius):
        mask = np.ones((self.oversamp, self.oversamp))
        
        # fill in the mask
        for nc in range(0, np.shape(center)[0]):
            mc = np.zeros_like(mask)
            mc_coords = draw.circle(center[nc][0], center[nc][1], radius=radius[nc])
            mc[mc_coords] = True
            mask[mc==True] = np.nan
        
        self.psd_raw *= mask
        self.psd_norm *= mask
        self.psd_cal *= mask
    
    def calc_psd_radial(self, ring_width, kmin=None):
        # shortcut version for basic code analysis
        (self.k_radial, self.psd_radial_cal) = do_psd_radial(psd_data=self.psd_cal, delta_k=self.delta_k, ring_width=ring_width, kmin=kmin)
    
    def calc_rms_set(self, kmid_ll, khigh_ll, pwr_opt, print_rms=False, print_kloc=False):
        # Calculate the RMS based on the k-parameter limits
        # all RMS units are same units as data and variance.
        self.kmid_ll = kmid_ll
        self.khigh_ll = khigh_ll
        self.rms_tot = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[self.k_min, self.k_max],
                                  print_rms=print_rms, print_kloc=print_kloc)
        self.rms_l = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[self.k_min, kmid_ll],
                                  print_rms=print_rms, print_kloc=print_kloc)
        self.rms_m = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[kmid_ll, khigh_ll],
                                  print_rms=print_rms, print_kloc=print_kloc)
        self.rms_h = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[khigh_ll, self.k_max],
                                  print_rms=print_rms, print_kloc=print_kloc)
        self.rms_mh = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[kmid_ll, self.k_max],
                                  print_rms=print_rms, print_kloc=print_kloc)
    
    def write_psd_file(self, filename, psd_data, single_precision=True):
        # Write header and cards for FITS
        hdr = fits.Header()
        hdr['name'] = (self.surf_name + ' PSD', 'filename')
        hdr['psd_unit'] = (str(psd_data.unit), 'Units for PSD data')
        hdr['wavelen'] = (self.wavelen.value, 'Wavelength used for optical test [{0}]'.format(self.wavelen.unit))
        hdr['diam_ca'] = (self.diam_ca.value, 'Physical diameter for clear aperture [{0}]'.format(self.diam_ca.unit))
        hdr['diam_pix'] = (self.npix_diam, 'Pixel diameter for clear aperture')
        hdr['oversamp'] = (self.oversamp, 'Oversampled array size')
        hdr['delta_k'] = (self.delta_k.value, 'Spatial frequency lateral resolution [{0}]'.format(self.delta_k.unit))
        hdr['k_min'] = (self.k_min.value, 'Minimum spatial frequency boundary [{0}]'.format(self.k_min.unit))
        hdr['k_max'] = (self.k_max.value, 'Maximum spatial frequency boundary [{0}]'.format(self.k_max.unit))
        hdr['rms_tot'] = (self.rms_tot.value, 'Total RMS based on kmin and kmax [{0}]'.format(self.rms_tot.unit))
        hdr['var_tot'] = (self.var.value, 'Total variance for optical surface [{0}]'.format(self.var.unit))
        
        if single_precision==True:
            write_data = np.single(psd_data.value)
        else:
            write_data = psd_data.value
            
        fits.writeto(filename, write_data, hdr, overwrite=True)

##########################################
# PSD SPECIAL CASE: 2-D LOMB-SCARGLE
def mvls_psd(data, mask, dx, k_side, print_update=False, 
             write_psd=False, psd_name=None, psd_folder=None):
    
    # do some quick checking before running the whole code
    if write_psd == True:
        if psd_name == None:
            raise Exception('Cannot write PSD to FITS file without psd_name passed in.')
        if psd_folder is not None:
            if os.path.isdir(psd_folder) is False:
                # check if it's a weird path, then fix it
                folder_dir = os.getcwd() + '/' + psd_folder
                if os.path.isdir(folder_dir) is True:
                    psd_folder = folder_dir
                else:
                    raise Exception('PSD folder destination bad; please check again')
    
    # create the vector array for spatial coordinates
    data_side = np.shape(data)[0]
    cen = int(data_side/2)
    if data_side % 2 == 0: # even width
        yy, xx = np.mgrid[-cen:cen, -cen:cen]
    else:
        yy, xx = np.mgrid[-cen:cen+1, -cen:cen+1]
    tnx = xx * dx.value # unitless
    tny = yy * dx.value # unitless
    
    # load the data and apply a window to it
    surf_win = data.value * han2d((data_side, data_side)) # unitless
    
    # filter the data using the mask
    mask_filter = np.where(mask==1) # automatically vectorizes the data
    tn = np.vstack((tnx[mask_filter], tny[mask_filter]))
    ytn = surf_win[mask_filter]
    
    # build the spatial frequency coordinates
    # k_side is not necessarily the same size as the data coming in.
    dk = 1/(data_side*dx.value) # unitless
    k_tot = k_side**2
    k_cen = int(k_side/2)
    if k_side % 2 == 0:
        ky, kx = np.mgrid[-k_cen:k_cen, -k_cen:k_cen]
    else:
        ky, kx = np.mgrid[-k_cen:k_cen+1, -k_cen:k_cen+1]
    kx = kx*dk
    ky = ky*dk
    wkx = np.reshape(kx, k_tot) # unitless
    wky = np.reshape(ky, k_tot) # unitless
    
    '''
    By this point, to do the Lomb-Scargle, the following should be unitless:
    tn, ytn, wkx, wky
    This will allow tau, ak, bk to be unitless.
    The units will all be reapplied at the end
    '''
    
    # initialize variables for the Lomb-Scargle
    tau = np.zeros((k_tot))
    ak = np.zeros((k_tot))
    bk = np.zeros((k_tot))
    
    # begin the scargling
    if print_update == True:
        print('Scargling in progress, starting time =', datetime.now().strftime("%H:%M:%S"))
    for nk in range(0, k_tot):
        # calculate the dot product
        wkvec = ([wkx[nk], wky[nk]])
        wdt = np.dot(wkvec, tn) * 2 * np.pi # mandatory 2pi for radians units
        
        # calculate tau
        tau_num = np.sum(np.cos(2*wdt))
        tau_denom = np.sum(np.sin(2*wdt))
        tau_val = 0.5 * np.arctan2(tau_num, tau_denom) # tau is radians here
        tau[nk] = tau_val # load into the tau array
        
        # calculate inner pdoruct for ak and bk
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
        
        # print out updates if requested
        if print_update == True:
            if nk % 10000 == 0:
                print('{0} of {1} complete ({2:.2f}%), time ='.format(nk, k_tot, 
                                                                      nk*100/k_tot)
                , datetime.now().strftime("%H:%M:%S"))
        
    # outside the loop, all of tau, ak, bk should be filled in
    # calculate the PSD
    psd = ((ak**2) + (bk**2)) / (dk**2)
    psd = np.reshape(psd, (k_side, k_side)) * (data.unit*dx.unit)**2
    if print_update == True:
        print('Scargling and PSD completed, ending time =', datetime.now().strftime("%H:%M:%S"))
    
    if write_psd==True: # write PSD to fits file if requested
        hdr = fits.Header()
        hdr['name'] = (psd_name, 'filename')
        hdr['psd_unit'] = (str(psd.unit), 'Units for PSD data')
        hdr['surfunit'] = (str(data.unit), 'Units used for surface data')
        hdr['latres'] = (dx.value, 'Data spatial resolution [{0}]'.format(dx.unit))
        hdr['diam_ca'] = (data_side*dx.value, 'Physical diameter for clear aperture [{0}]'.format(dx.unit))
        hdr['diam_pix'] = (data_side, 'Pixel diameter for data clear aperture')
        hdr['delta_k'] = (dk, 'Spatial frequency lateral resolution [1/{0}]'.format(dx.unit))
        
        if psd_folder is not None:
            psd_filename = psd_folder+psd_name+'.fits'
        else:
            psd_filename = psd_name+'.fits'
        
        fits.writeto(psd_filename, psd.value, hdr, overwrite=True)
    
    # apply all the units
    lspsd_parms = {'dk': dk/dx.unit,
                   'radialFreq': kx[0]/dx.unit}
    
    return psd, lspsd_parms
    
##########################################
# MODELING
'''
Separate from the main class because this is applied only to the average PSD.
Class is applied to individual surface PSD.

Assumptions on the units:
alpha           - unitless
beta            - nm^2 mm^(-alpha+2). It washes out at the end.
L0              - mm (1/spatial frequency = (1/(1/mm) = mm)
lo              - unitless
k_min, k_max    - 1/mm (spatial frequency)
rms_sr          - nm (surface roughness)
rms_surf        - nm (total surface)
radial_k        - 1/mm (spatial frequency)
radial_psd      - mm^2 nm^2 (1/spatial frequency ^2 * variance = mm^2 nm^2)
bsr        - mm^2 nm^2 (same units as psd)
'''
class model_single(surfPSD):
    def __init__(self, region_num):#, ind_start, ind_end, surfPSD):
        self.region_num = region_num
    
    def set_data(self, ind_range, k_radial, p_radial, k_min, k_max):
        if hasattr(k_radial, 'unit') and hasattr(p_radial, 'unit'):
            self.k_radial = k_radial
            self.p_radial = p_radial
            self.k_data = k_radial[ind_range[0]:ind_range[1]+1]
            self.p_data = p_radial[ind_range[0]:ind_range[1]+1]
            self.surf_unit = (p_radial.unit*(k_radial.unit**2))**(0.5) # sqrt alternative
        else:
            raise Exception('k-space and/or PSD data need units.')
        
        if hasattr(k_min, 'unit'):
            if (k_min.unit != k_radial.unit):
                self.k_min = k_min.to(k_radial.unit)
            else:
                self.k_min = k_min
        else:
            raise Exception('k_min needs units and preferably should match with k_radial')
        if hasattr(k_max, 'unit'):
            if (k_max.unit != k_radial.unit):
                self.k_max = k_max.to(k_radial.unit)
            else:
                self.k_max = k_max
        else:
            raise Exception('k_max needs units and preferably should match with k_radial')
    
    def load_data(self, ind_range, psd_obj=None):
        self.i_start = ind_range[0]
        self.i_end = ind_range[1]
        if psd_obj is not None: # passed in a separate object
            self.k_data = psd_obj.k_radial[ind_range[0]:ind_range[1]]
            self.p_data = psd_obj.psd_radial_cal[ind_range[0]:ind_range[1]]
            self.k_min = psd_obj.k_min
            self.k_max = psd_obj.k_max
        else: # surfPSD methods were used and was calculated
            self.k_data = self.k_radial[ind_range[0]:ind_range[1]]
            self.p_data = self.psd_radial_cal[ind_range[0]:ind_range[1]]
            # self.k_min and self.k_max not needed because pre-existing
        self.surf_unit = (self.p_data.unit*(self.k_data.unit**2))**(0.5) # sqrt alternative
        
    def load_psd_parm(self, psd_parm_list):
        self.alpha = psd_parm_list[0]
        self.beta = psd_parm_list[1]
        self.L0 = psd_parm_list[2]
        self.lo = psd_parm_list[3]
        self.bsr = psd_parm_list[4]
        
    def calc_psd_parm(self, rms_sr, x0=[1.0, 1.0, 1.0, 1.0]):
        # calculate the surface roughness value
        if hasattr(rms_sr, 'unit'): # unit check and fix
            if rms_sr.unit != self.surf_unit:  
                print(rms_sr.unit, self.surf_unit)
                rms_sr.to(self.surf_unit)
            self.rms_sr = rms_sr
        else:
            raise Exception('surface roughness RMS needs units')
        self.bsr = model_bsr(k_min=self.k_min, k_max=self.k_max, rms_sr=self.rms_sr)
        
        # calculate the rest of the variables
        # x0 are guessing values for non-linear lsq, they can be anything.
        res_lsq = least_squares(fit_func, x0, args=(self.k_data.value, self.p_data.value))
        
        self.alpha = res_lsq.x[0]
        self.beta = res_lsq.x[1] * (self.surf_unit**2) / (self.k_data.unit**(-self.alpha+2))
        self.L0 = res_lsq.x[2]/self.k_data.unit
        self.lo = res_lsq.x[3]
        
    def calc_model_total(self, psd_weight=1.0, k_range=None, k_spacing=None, k_limit=None):
        # set the k-range
        if k_range is None: # k_range passed in as None defaults to setting k_range
            if k_spacing is None:
                raise Exception('Need spatial frequency spacing to build k-space')
            if k_limit is None:
                raise Exception('Need upper and lower bound limits for spatial frequency')
            k_range = set_k_range(k_spacing=k_spacing, k_limit=k_limit)
        self.k_range = k_range
        
        # verify the PSD units will match with bsr's unit before moving forward
        pmdl_unit_0 = self.beta.unit/(self.L0.unit**-self.alpha)
        pmdl_unit_1 = self.beta.unit/(self.k_data.unit**self.alpha)
        if (pmdl_unit_0 != self.bsr.unit) or (pmdl_unit_1 != self.bsr.unit):
            raise Exception('PSD units not matching with Beta_sr units, something is wrong somewhere.')
        else: # units matching, move forward
            self.psd_weight = psd_weight
            psd_parm=[self.alpha, self.beta, self.L0, self.lo, self.bsr]
            self.psd_full=model_full(k=k_range, psd_parm=psd_parm)
            self.psd_full_scaled = self.psd_full * psd_weight
    
    def calc_beta(self, alpha, rms_surf):
        # unit check and fix
        if hasattr(rms_surf, 'unit'):
            if rms_surf.unit != self.surf_unit:   
                rms_surf.to(self.surf_unit)
        else:
            raise Exception('Surface RMS needs units')
        # calculate beta
        self.beta_calc = model_beta(k_min=self.k_min, k_max=self.k_max,
                                    alpha=alpha, rms_surf=rms_surf)

# Supporting functions inside the model class
def psd_fitter(x, k):
    # x: parameters to fit
    # k: spatial frequency values
    n_prm = 4 # THERE ARE FOUR LIGHTS -Captain Jean-Luc Picard
    n_psd = int(len(x)/n_prm)
    for j in range(0, n_psd):
        i_offset = int(j*n_prm)
        denom = ( ((1/x[2+i_offset])**2) + (k**2))**(x[0+i_offset]/2)
        if j== 0:
            pk = (x[1+i_offset]/denom)*np.exp(-((k*x[3+i_offset])**2))
        else:
            pk = pk + (x[1+i_offset]/denom)*np.exp(-((k*x[3+i_offset])**2))
    return pk

def fit_func(x, k, y):
    # y: values to fit against
    pk = psd_fitter(x,k)
    return pk-y

def fit_func_ratio(x, k, y):
    return fit_func(x,k,y)/y

def model_beta(k_min, k_max, alpha, rms_surf):
    if alpha==2:
        beta = (rms_surf**2) / (2*np.pi*np.log(k_max/k_min))
    else: # when alpha not 2
        beta = (rms_surf**2)*(alpha-2) / (2*np.pi*( (k_min**(2-alpha)) - (k_max**(2-alpha))))
    return beta # units safe

def model_bsr(k_min, k_max, rms_sr):
    return (rms_sr**2) / (np.pi * (k_max**2 - k_min**2)) # units safe

def model_full(k, psd_parm):
    # calculates a single power instance based on passed parameters
    alpha=psd_parm[0]
    beta=psd_parm[1]
    L0=psd_parm[2]
    lo=psd_parm[3]
    bsr=psd_parm[4]
    # verify the L0 value before passing it through
    if L0.value == 0: # need to skip out the L0 value or else it explodes
        pk = beta/((k**2)**(alpha*.5))
    else:
        if L0.unit != (1/k.unit):# check units before calculating
            print('Changing L0 unit to match with 1/dk units')
            L0.to(1/k.unit) # match the unit with the spatial frequency
        pk = beta / (((L0**-2) + (k**2))**(alpha*.5))
    if lo != 0: #lo should be a unitless number
        pk = pk * np.exp(-(k.value*lo)**2) # the exponential needs to be unitless
    if hasattr(bsr, 'unit'): # if there are units in bsr, convert them to the correct value.
        if bsr.unit != pk.unit:
            bsr.to(pk.unit)
        if bsr.value != 0:
            pk = pk + bsr
    else:
        raise Exception('Beta_sr lacks units')
    # check for infinity presence (where k=0 and L0=0), then override it to 0.
    inf_coord = np.argwhere(pk.value==np.inf)
    if inf_coord.shape[0]>0:
        pk[inf_coord.tolist()[0][0]][inf_coord.tolist()[0][1]] = 0*pk.unit
    return pk # will have units

###########################################
# MODEL APPLICATION

class model_combine:
    def __init__(self, mdl_set, avg_psd):
        # collect data from the average PSD object
        self.delta_k = avg_psd.delta_k
        self.npix_diam = avg_psd.npix_diam
        self.k_min = avg_psd.k_min
        self.k_max = avg_psd.k_max
        self.k_radial_data = avg_psd.k_radial
        self.psd_radial_data = avg_psd.psd_radial_cal
        
        # collect data from the single PSD models
        psd_parm = []
        psd_weight = []
        sum_mdl = np.zeros_like(mdl_set[0].psd_full.value)
        sum_mdl_data = np.zeros_like(self.psd_radial_data.value)
        for j in range(0, len(mdl_set)):
            parameters = [mdl_set[j].alpha, mdl_set[j].beta, mdl_set[j].L0, mdl_set[j].lo, mdl_set[j].bsr]
            psd_parm.append(parameters)
            psd_weight.append(mdl_set[j].psd_weight)
            sum_mdl = sum_mdl + mdl_set[j].psd_full_scaled.value
            sum_mdl_data = sum_mdl_data + model_full(k=self.k_radial_data, psd_parm=parameters).value
        self.psd_parm = psd_parm
        self.psd_weight = psd_weight
        self.psd_radial_sum = sum_mdl * mdl_set[0].psd_full.unit
        self.psd_radial_sum_data = sum_mdl_data * mdl_set[0].psd_full.unit # should have same units
        self.k_radial_model = mdl_set[0].k_range
        self.surf_unit = mdl_set[0].surf_unit
        
    def calc_refit(self):
        x0 = []
        n_psd = len(self.psd_parm)
        for j in range(0, n_psd):
            x0.append(self.psd_parm[j][0])
            x0.append(self.psd_parm[j][1].value)
            x0.append(self.psd_parm[j][2].value)
            x0.append(self.psd_parm[j][3])
        full_lsq = least_squares(fit_func_ratio, x0, args=(self.k_radial_data.value, self.psd_radial_data.value))
        
        # reset the psd parameters and the total PSD sums
        new_parm = []
        mdl_sum = np.zeros_like(self.psd_radial_sum.value)
        mdl_sum_data = np.zeros_like(self.psd_radial_data.value)
        for j in range(0, n_psd):
            i_offset = int(j*4)
            a = full_lsq.x[0+i_offset]
            b = full_lsq.x[1+i_offset] * (self.surf_unit**2) / (self.k_radial_data.unit**(-a+2))
            os = full_lsq.x[2+i_offset]/self.k_radial_data.unit
            ins = full_lsq.x[3+i_offset]
            parm = [a, b, os, ins, self.psd_parm[j][4]]
            new_parm.append(parm)
            new_mdl = model_full(k=self.k_radial_model, psd_parm=parm)
            mdl_sum = mdl_sum + (new_mdl.value * self.psd_weight[j])
            mdl_sum_data = mdl_sum_data + (model_full(k=self.k_radial_data, psd_parm=parm).value * self.psd_weight[j])
        self.psd_parm = new_parm
        self.psd_radial_sum = mdl_sum * new_mdl.unit
        self.psd_radial_sum_data = mdl_sum_data * new_mdl.unit 
        
    def overwrite_parm(self, section_num, overwrite_parm):
        n_psd = len(self.psd_parm)
        new_parm = copy.copy(self.psd_parm)
        new_parm[section_num] = overwrite_parm
        
        # reset the psd parameters and the total PSD sums
        mdl_sum = np.zeros_like(self.psd_radial_sum.value)
        mdl_sum_data = np.zeros_like(self.psd_radial_data.value)
        for j in range(0, n_psd):
            new_mdl = model_full(k=self.k_radial_model, psd_parm=new_parm[j])
            mdl_sum = mdl_sum + (new_mdl.value * self.psd_weight[j])
            mdl_sum_data = mdl_sum_data + (model_full(k=self.k_radial_data, psd_parm=new_parm[j]).value * self.psd_weight[j])
        self.psd_parm = new_parm
        self.psd_radial_sum = mdl_sum * new_mdl.unit
        self.psd_radial_sum_data = mdl_sum_data * new_mdl.unit 
        
        
    def calc_error(self):
        self.error = np.log10(self.psd_radial_sum_data.value/self.psd_radial_data.value)
        self.error_rms = np.sqrt(np.mean(np.square(self.error)))
        
    def calc_psd_rms(self):
        # build the k-map
        cen = int(self.npix_diam/2)
        if self.npix_diam%2 == 0:
            ky, kx = np.ogrid[-cen:cen, -cen:cen]
        else:
            ky, kx = np.ogrid[-cen:cen+1, -cen:cen+1]
        ky = ky*self.delta_k
        kx = kx*self.delta_k
        k_map = np.sqrt(kx**2 + ky**2)
        
        # calculate the 2D PSD
        psd_mdl = np.zeros_like(k_map.value)
        for n in range(0, len(self.psd_weight)):
            psd_mdl = psd_mdl + (model_full(k=k_map, psd_parm=self.psd_parm[n]).value * self.psd_weight[n])
        psd_2D = psd_mdl * self.psd_radial_sum.unit
        
        # with the PSD calculated, do the RMS
        k_tgt_lim = [self.k_min, self.k_max]
        self.psd_rms_sum = do_psd_rms(psd_data=psd_2D, delta_k=self.delta_k, 
                                  k_tgt_lim=k_tgt_lim, print_rms=False, print_kloc=False)

# Plotting assistance
def plot_model2(mdl_set, model_sum, avg_psd, opt_parms):
    k_radial = avg_psd.k_radial.value
    psd_radial = avg_psd.psd_radial_cal.value
    k_range_mdl = mdl_set[0].k_range.value
    
    color_list=['r', 'b', 'y', 'g', 'c']
    anno_opts = dict(xy=(0.1, .9), xycoords='axes fraction',
                     va='center', ha='center')

    plt.figure(figsize=[14,9],dpi=100)
    gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4,2])
    ax0 = plt.subplot(gs[0])  
    ax0.loglog(k_radial, psd_radial, 'k', linewidth=3, label='Avg PSD (PTT, all steps)\nRMS={0:.5f}'.format(avg_psd.rms_tot))
    for j in range(0, len(mdl_set)):
        psd_value = model_full(k=mdl_set[0].k_range, psd_parm=model_sum.psd_parm[j]) * model_sum.psd_weight[j]
        plt_label = '{0}: {1}={2:.3f} {3}={4:.3e}\n'.format(r'$r_{0}$'.format(mdl_set[j].region_num),
                                                             r'$\alpha$', model_sum.psd_parm[j][0], 
                                                             r'$\beta$', model_sum.psd_parm[j][1].value)
        plt_label = plt_label + '{0}={1:.1e}, {2}={3:.2e}, {4}={5:.1e}A\n'.format(r'$L_{0}$',model_sum.psd_parm[j][2], 
                                                                                   r'$l_{0}$', model_sum.psd_parm[j][3], 
                                                                                   r'$\sigma_{sr}$', mdl_set[j].rms_sr.to(u.angstrom).value)
        plt_label = plt_label + '{0}={1:.2f}'.format(r'$a_{0}$'.format(j), model_sum.psd_weight[j])
        ax0.loglog(mdl_set[j].k_range.value, psd_value.value, color_list[j]+':', linewidth=1.5,
                   label=plt_label)
        # draw in the color box
        ax0.axvspan(k_radial[mdl_set[j].i_start], k_radial[mdl_set[j].i_end], facecolor=color_list[j], alpha=0.1)
    mdl_sum_text = 'model sum {0}\nRMS={1:.5f}'.format(r'$\Sigma a_{n}r_{n}$', model_sum.psd_rms_sum)
    ax0.loglog(k_range_mdl, model_sum.psd_radial_sum.value, linewidth=2.5, label=mdl_sum_text)
    ax0.set_xlim(left=np.amin(k_range_mdl)*0.8, right=np.amax(k_range_mdl)*1.2)
    ax0.set_ylim(bottom=1e-11)
    ax0.set_ylabel('PSD ({0})'.format(mdl_set[0].psd_full.unit))
    ax0.legend(prop={'size':8})#,loc='center left', bbox_to_anchor=(1, 0.5))
    ax0.set_title('MagAO-X PSD modeling (PRELIMINARY VISUAL): {0}, {1}% CA'.format(opt_parms['label'], opt_parms['ca']))
    
    err_rms = np.sqrt(np.mean(np.square(model_sum.error)))
    ax1 = plt.subplot(gs[1])
    ax1.semilogx(k_radial, model_sum.error)
    ax1.hlines(y=0, xmin=np.amin(k_range_mdl)*0.8, xmax=np.amax(k_range_mdl)*1.2, color='k')
    ax1.set_ylim(top=0.5, bottom=-0.5)
    ax1.set_xlim(left=np.amin(k_range_mdl)*0.8, right=np.amax(k_range_mdl)*1.2)
    ax1.set_ylabel('log10( model / measured)')
    ax1.set_xlabel('Spatial Frequency [{0}]'.format(mdl_set[0].k_range.unit))
    ax1.annotate('Error RMS: {0:.4}'.format(err_rms), **anno_opts)

    plt.tight_layout()


###########################################
# BUILD SURFACE
'''
NOTE: This section is getting imported to POPPY eventually. Kept here for test coding.
'''
class surfgen:
    def __init__(self, dx, npix_diam, oversamp):
        self.dx = dx # phase space resolution [m/pix]
        self.npix_diam = npix_diam # non-zero-padded diameter for POPPY
        self.oversamp = oversamp # zero padded array size for FT
        self.dk = 1/(oversamp * dx) # target fourier space resolution
            
    def calc_psd(self, psd_parm, psd_weight):
        self.psd_parm=psd_parm
        self.psd_weight=psd_weight
        # build k-space map
        cen = int(self.oversamp/2)
        maskY, maskX = np.ogrid[-cen:cen, -cen:cen]
        ky = maskY*self.dk
        kx = maskX*self.dk
        self.k_map = np.sqrt(kx**2 + ky**2)
        
        # initialize the empty matrices
        psd_map = np.zeros((self.oversamp, self.oversamp))
        for nm in range(0, len(psd_weight)):
            psd_map = psd_map + (psd_weight[nm] * model_full(k=self.k_map, psd_parm=psd_parm[nm]))
        self.psd=psd_map # should clean up units
        
    def init_noise(self, mean=0,std=1):
        self.noise_ph = np.random.normal(mean, std, (self.oversamp, self.oversamp))
        self.noise_ft = np.fft.fftshift(np.fft.fft2(self.noise_ph))
        
    def build_surf(self, noise_mean=0, noise_std=1, opd_reflect=True): # undergoing work, do not use this.
        self.opd_reflect=opd_reflect
        self.init_noise(mean=noise_mean, std=noise_std)
        psd_sqrt=np.sqrt(self.psd/(self.dx**2))
        cor_noise = np.fft.ifft2(np.fft.ifftshift(self.noise_ft * psd_sqrt))*psd_sqrt.unit
        surface_full = np.real(cor_noise)
        if opd_reflect==True:
            surface_full = surface_full/2 # divide by 2 for reflection WFE
        self.surface = dfx.doCenterCrop(optic_data=surface_full, shift=int(self.npix_diam/2))
        self.rms = np.sqrt(np.sum(self.surface**2)/(self.npix_diam**2))
        
    #def write_psd_file(self, filename, psd_data, single_precision=True):
    def write_surf_file(self, file_folder, filename, single_precision=False):
        # Write header and cards for FITS
        hdr = fits.Header()
        #hdr['name'] = (self.surf_name + ' PSD', 'filename')
        hdr['opd_unit'] = (str(self.surface.unit), 'Units for surface OPD data')
        hdr['opd_refl'] = (self.opd_reflect, 'Boolean if surface OPD/2 for reflection')
        hdr['op_diam'] = (self.npix_diam*self.dx.value, 'Physical diameter for clear aperture [{0}]'.format(self.dx.unit))
        hdr['scrnsz'] = (self.oversamp, 'Array size for screen generation')
        hdr['pixscale'] = (self.dx.value, 'pixel scale [{0}/pix]'.format(self.dx.unit))
        hdr['delta_k'] = (self.dk.value, 'Spatial frequency lateral resolution [{0}]'.format(self.dk.unit))
        for n in range(0, len(self.psd_weight)):
            hdr['alpha{0}'.format(n)] = (self.psd_parm[n][0], 'PSD alpha (exponent)')
            hdr['beta{0}'.format(n)] = (self.psd_parm[n][1].value, 
                                        'PSD beta (normalization) [{0}^2 {1}^(2-alpha)]'.format(self.surface.unit, self.dx.unit))
            hdr['OS{0}'.format(n)] = (self.psd_parm[n][2].value, 
                                      'PSD L0 (outer scale) [{0}]'.format(self.psd_parm[n][2].unit))
            hdr['IS{0}'.format(n)] = (self.psd_parm[n][3], 'PSD l0 (inner scale)')
            hdr['beta_sr{0}'.format(n)] = (self.psd_parm[n][4].value, 
                                           'PSD surface roughness normalization [{0}^2 {1}^2]'.format(self.surface.unit, self.dx.unit))
            hdr['weight{0}'.format(n)] = (self.psd_weight[n], 'Applied PSD weight value')
        if single_precision==True:
            write_data = np.single(self.surface.value)
        else:
            write_data = self.surface.value
        
        # Write to FITS file
        fits.writeto(file_folder+filename, write_data, hdr, overwrite=True)
        
        
###########################################
# INTERPOLATION

def k_interp(oap_label, kval, npix_diam, norm_1Dpsd, cal_1Dpsd, k_npts):
    kmin = []
    kmax = []
    ntot = np.shape(npix_diam)[0]
    
    # calculate the min and max
    for no in range(0,ntot):
        oap_name = oap_label[no]
        kmax.append(np.amax(kval[oap_name]).value)
        kmin.append(np.amin(kval[oap_name]).value)
        
    kmin_interp = np.amax(kmin)*kval[oap_name].unit
    kmax_interp = np.amin(kmax)*kval[oap_name].unit
    print('kmin = {0:.4f}, kmax = {1:.4f}'.format(kmin_interp, kmax_interp))
    
    # check if interpolation needs to happen:
    if (np.unique(npix_diam).size==1) and (np.unique(kmax).size==1) and (np.unique(kmin).size==1):
        psd1d_interp = False
        print('1D-PSD interpolation does not need to occur; #pts of spatial freq: {0}'.format(k_npts))
        
    else:
        psd1d_interp = True
        k_new = np.linspace(kmin_interp, kmax_interp, num=k_npts) # new range of spatial frequencies
        print('1D-PSD interpolation does needs to occur; #pts of spatial freq: {0}'.format(np.shape(k_new)[0]))

    # Write the matrices for the data
    psd1D_data = np.zeros((ntot, k_npts)) # all k_val is same size
    psd1D_norm = np.zeros((ntot, k_npts)) # all k_val is same size
    k_out = np.zeros_like((psd1D_data))
    
    # fill in the data matricies from the PSD simulation and interpolate
    for no in range(0,ntot):
        oap_name = oap_label[no]
        k_orig = kval[oap_name]
        psd_data = cal_1Dpsd[oap_name]
        psd_norm = norm_1Dpsd[oap_name]

        if psd1d_interp == True:
            # calculate the interpolation function based on the data
            f_data = interpolate.interp1d(k_orig, psd_data)
            f_norm = interpolate.interp1d(k_orig, psd_norm)
            # fill in the interpolation for specific spatial frequencies
            psd1D_data[no,:] = f_data(k_new)
            psd1D_norm[no,:] = f_norm(k_new)
            k_out[no, :] = k_new
            
        else:
            psd1D_data[no, :] = psd_data
            psd1D_norm[no, :] = psd_norm
            k_out[no, :] = k_orig

    # apply units
    psd1D_data *= cal_1Dpsd[oap_name].unit
    psd1D_norm *= norm_1Dpsd[oap_name].unit
    
    return(psd1D_data, psd1D_norm, k_out)


###########################################
# SUPPORT

def han2d(shape, fraction=1./np.sqrt(2), normalize=False):
    '''
    Code written by Kyle Van Gorkom.
    Radial Hanning window scaled to a fraction 
    of the array size.
    
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''
    #return np.sqrt(np.outer(np.hanning(shape[0]), np.hanning(shape[0])))

    # get radial distance from center
    radial = get_radial_dist(shape)

    # scale radial distances
    rmax = radial.max() * fraction
    scaled = (1 - radial / rmax) * np.pi/2.
    window = np.sin(scaled)**2
    window[radial > fraction * radial.max()] = 0.
    return window

def get_radial_dist(shape, scaleyx=(1.0, 1.0)):
    '''
    Code written by Kyle Van Gorkom.
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def set_k_range(k_spacing, k_limit):
    # custom setting spatial frequency range
    k_extend = np.arange(start=k_limit[0].value, stop=k_limit[1].value, step=k_spacing.value)
    return k_extend * k_spacing.unit

def do_psd_radial(psd_data, delta_k, ring_width=3, kmin=None):
    # a different version for less frills.
    # formerly new_psd_radial
    if hasattr(psd_data, 'unit'):
        psd_unit = psd_data.unit
        psd_data = psd_data.value
    else:
        raise Exception('PSD missing units')
    
    if hasattr(delta_k, 'unit'):
        delta_k_unit = delta_k.unit
        delta_k = delta_k.value
    else:
        raise Exception('Radial frequency scale missing units')
    
    if ring_width % 2 != 0: # must be odd
        ring_side = int((ring_width-1)/2)
    else: # even width, not okay
        raise Exception('Ring width needs to be odd for symmetric fit')
    
    side = np.shape(psd_data)[0]
    cen = int(side/2)
    if side%2 == 0:
        my, mx = np.ogrid[-cen:cen, -cen:cen]
    else:
        my, mx = np.ogrid[-cen:cen+1, -cen:cen+1]
    radial_freq = mx[0][cen:side]*delta_k
    
    if kmin is not None:
        ind_start, k_val = k_locate(freqrange = radial_freq * delta_k_unit, k_tgt = kmin)
        if k_val < kmin:
            ind_start = ind_start + 1
    else:
        ind_start = ring_side

    # initialize content
    mean_bin = [] # initialize empty list of mean PSD values
    k_val = [] # initialize empty list of spatial frequencies
    
    #for test_radius in range(ring_side, cen): # starting at ring_side provides minimum distance for calculation
    for test_radius in range(ind_start, cen):
        ring_mask_in = mx**2 + my**2 <= (test_radius-ring_side)**2
        ring_mask_out = mx**2 + my**2 <= (test_radius+ring_side)**2
        ring_mask = ring_mask_out ^ ring_mask_in #XOR statement
        ring_bin = np.extract(ring_mask, psd_data)
        mean_bin.append(np.mean(ring_bin))
        k_val.append(radial_freq[test_radius])

    k_radial = np.asarray(k_val)*delta_k_unit
    psd_radial = mean_bin * psd_unit #* psd_data.unit

    return (k_radial, psd_radial) # both should hold units

def do_psd_rms(psd_data, delta_k, k_tgt_lim, print_rms=False, print_kloc=False):   
    # create the grid
    side = np.shape(psd_data)[0]
    cen = int(side/2)
    if side%2 == 0:
        my, mx = np.ogrid[-cen:cen, -cen:cen]
    else:
        my, mx = np.ogrid[-cen:cen+1, -cen:cen+1]
    radial_freq = mx[0][cen:side]*delta_k
    
    # find the locations for k_low and k_high:
    (bin_low, k_low) = k_locate(radial_freq, k_tgt_lim[0], print_change=print_kloc)
    (bin_high, k_high) = k_locate(radial_freq, k_tgt_lim[1], print_change=print_kloc)
    rms_width = bin_high - bin_low
    
    # make the mask
    ring_mask_in = mx**2 + my**2 <= bin_low**2
    ring_mask_out = mx**2 + my**2 <= bin_high**2
    ring_mask = ring_mask_out ^ ring_mask_in #XOR statement
    ring_bin = np.extract(ring_mask, psd_data) #* psd_data.unit
    
    # calculate the rms
    rms_val = np.sqrt(np.sum(ring_bin * (delta_k**2))) # should have units
    if print_rms==True:
        print('Target range - k_low: {0:.3f} and k_high: {1:.3f}'.format(k_low, k_high))
        print('RMS value: {0:.4f}'.format(rms_val))
    return rms_val

def k_locate(freqrange, k_tgt, print_change=False):
    # given a target spatial frequency, find the index bin and value closest to target.
    kscale = np.abs(freqrange.value - k_tgt.value)
    bb = np.where(kscale==np.amin(kscale))
    if freqrange[bb][0] != k_tgt and print_change == True:
        print('Target: {0:.4f}; changing to closest at {1:.4f}'.format(k_tgt, freqrange[bb][0]))
    return (bb[0][0], freqrange[bb][0])

# verbatim taken from numpy.pad website example
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder',0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
