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
from . import psd_wfe_poppy

##########################################
# surfPSD Class definition

class surfPSD:
    '''
    Class for creating the PSD for a surface measurement.
    Used in MagAO-X, CDEEP, and CACTI.
    '''
    
    def __init__(self, surf_name, **kwargs):
        self.surf_name = surf_name
        
    def open_surf(self, fileloc, surf_units):
        '''
        Use this function if opening data from a FITS file
        Assumes data is efficienty filled (no extra row/col with only zeros)
        Parameters:
            fileloc: string
                Filename of the surface file to be opened.
                Must include the '.fits' portion in the file.
            surf_units: astropy units
                Surface height units provided by the Zygo or whatever you want.
                Preferred: nanometers or microns (Zygo)
        '''
        self.data = (fits.open(fileloc)[0].data*surf_units) # units from Zgyo analysis
        hdr = fits.open(fileloc)[0].header
        self.wavelen = hdr['WAVELEN'] * u.m
        self.latres = hdr['LATRES'] * u.m / u.pix
    
    def load_surf(self, data, wavelen, latres):
        '''
        Use this function if the data is already loaded into an environment
        Parameters:
            data: numpy array with astropy units
                Surface data with units of height.
                Data must be efficiently filled (no extra row/col with only zeros)
            wavelen: float with astropy units
                Wavelength of beam used for measurement
                Units: meters
            latres: float with astropy units
                Spatial lateral resolution of data
                Assumes 1:1 aspect ratio for vertical and horizontal
                Units: meters
        '''
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
        '''
        Opens the surface mask from a file.
        Assumes data is efficienty filled (no extra row/col with only zeros)
        Parameters:
            fileloc: string
                Filename of the mask file to be opened.
                Must include the '.fits' portion in the file.
        '''
        mask = fits.open(fileloc)[0].data
        self.load_mask(mask)
    
    def load_mask(self, mask):
        '''
        Loads the surface mask from the environment.
        Assumes data is efficienty filled (no extra row/col with only zeros)
        Parameters:
            mask: numpy array
                binary mask file of active region of surface.
        '''
        if mask.shape != self.data.shape:
            raise Exception('Mask and data are not compatiable (shape)')
        else:
            self.mask = mask.astype(bool)
            self.npix_diam = int(np.sum(mask[int(mask.shape[0]/2)]))
            self.diam_ca = (self.npix_diam * u.pix * self.latres)#.to(u.mm)
            
    def open_psd(self, psd_fileloc, psd_type, var_unit = u.nm):
        '''
        Opens the 2D PSD from a file.
        Parameters:
            psd_fileloc: string
                Filename of the mask file to be opened.
                Must include the '.fits' portion in the file.
            psd_type: string
                PSD type passed in.
                Has 2 options:
                    norm: normalized PSD
                    cal: variance-calibrated PSD
            var_unit: astropy units
                variance unit. Will be squared for actual variance.
                Default: nanometers
        '''
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
        '''
        Loads the 2D PSD locally from the environment.
        Parameters:
            psd_data: numpy
                2D PSD data, should have units
                Recommended: nm2 m2
            psd_type: string
                PSD type passed in.
                Has 3 options:
                    norm: normalized PSD
                    cal: variance-calibrated PSD
                    raw: uncalibrated PSD
            var_unit: astropy units
                variance unit. Will be squared for actual variance.
                Default: optional
        '''
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
        '''
        Loads the PSD parameters.
        Use this if using load_psd.
        Parameters:
            diam_ca: float with astropy units
                diameter of clear aperture surface.
                Units recommend: meters 
            npix_diam: integer
                number of pixels across the clear aperture diameter
            wavelen: float with astropy units
                Wavelength of beam used for measurement
                Units: meters
            delta_k: float with astropy units
                spatial frequency spacing.
                Optional, because can be back solved.
                Units: 1/m
            oversamp: integer
                Size of array from oversampling.
                Option, because can be back solved.
                Usually: 4096
        '''
        self.diam_ca = diam_ca
        self.npix_diam = npix_diam
        self.wavelen = wavelen
        if oversamp is not None:
            self.oversamp = oversamp
        else:
            self.oversamp = np.shape(self.psd_cal.value)[0]
        self.calc_psd_parameters(delta_k=delta_k) # calculate other necessary parameters
    
    def calc_psd_parameters(self, delta_k=None):
        '''
        Calculates the PSD parameters.
        Parameters:
            delta_k: float with astropy units
                spatial frequency spacing.
                Optional, because can be back solved.
                Units: 1/m
        '''
        self.k_min = 1/self.diam_ca
        self.k_max = 1/(2*self.diam_ca / self.npix_diam)
        if delta_k is not None:
            self.delta_k = delta_k
        else:
            self.delta_k = 1/(self.oversamp*self.diam_ca/self.npix_diam)
        
    def calc_psd(self, oversamp, kmid_ll = 0.1/u.mm, khigh_ll=1/u.mm, var_unit = u.nm):
        '''
        Calculates the 2D PSD. This is one of the workhorse functions.
        Parameters:
            oversamp: integer
                Size of array from oversampling.
                Option, because can be back solved.
                Usually: 4096
            kmid_ll: float with astropy unit
                Lower limit of the mid-range spatial frequency region.
                Technically optional unless looking for RMS at mid-spatial frequency.
                Can be left alone, but if code breaks just adjust to a lower number.
                Default: 0.1/u.mm
            khigh_ll: float with astropy unit
                Lower limit of the high-range spatial frequency region.
                Technically optional unless looking for RMS at high-spatial frequency.
                Can be left alone, but if code breaks just adjust to a lower number.
                Default: 1/u.mm
            var unit: astropy units
                Unit setting for surface height variance scaling.
                Defaults as nanometer, but can be anything.
        '''
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
        '''
        Verifies that the 2D PSD is normalized.
        Was made to be independent of setting it as a self.psd_norm
        Parameters:
            psd_norm: numpy array with units
                Normalized 2D PSD data
                units: [self.surf.units]**2
        '''
        var_verify = np.sum(psd_norm) * (self.delta_k**2) # unitless and 1
        psd_verify = np.allclose(1, var_verify)
        if psd_verify==True:
            print('PSD normalized: var={0:.3f}'.format(var_verify))
        else:
            print('PSD not normalized: var={0:.3f}. What happened?'.format(var_verify))
    
    def mask_psd(self, center, radius):
        '''
        Masks out regions in the 2D PSD.
        This was necessary for OAP-2 for MagAO-X.
        Parameters:
            center: numpy array
                Center pixel coordinates of regions to mask out.
                Each row has 2 values (row, column)
                Each row is a different center coordinate location
            radius: integer
                Radius value of masked region
        '''
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
        '''
        Calculates the radial PSD distribution via rms of annular regions.
        Parameters:
            ring_width: integer
                Width of ring for annulus calculation.
                Must be an odd number greater than 1.
            kmin: float (with units? I am not sure)
                Starting spatial frequency value for radial PSD calculation.
                Defaults to None, which starts at first opportunity in function.
        '''
        # shortcut version for basic code analysis
        (self.k_radial, self.psd_radial_cal) = do_psd_radial(psd_data=self.psd_cal, delta_k=self.delta_k, ring_width=ring_width, kmin=kmin)
    
    def calc_rms_set(self, kmid_ll, khigh_ll, pwr_opt, print_rms=False, print_kloc=False):
        '''
        Calculate the RMS of regions based on the k-parameter limits
        All RMS units are same units as data and variance.
        Parameters:
            kmid_ll: float with astropy unit
                Lower limit of the mid-range spatial frequency region.
                Technically optional unless looking for RMS at mid-spatial frequency.
                Can be left alone, but if code breaks just adjust to a lower number.
                Default: 0.1/u.mm
            khigh_ll: float with astropy unit
                Lower limit of the high-range spatial frequency region.
                Technically optional unless looking for RMS at high-spatial frequency.
                Can be left alone, but if code breaks just adjust to a lower number.
                Default: 1/u.mm
            pwr_opt: numpy array with astropy units
                2D PSD array with units for calculating RMS values.
                Good options: self.psd_cal
            print_rms: Boolean
                Determine whether to print out the rms values
                Default to False (do not print out the values)
        '''
        self.kmid_ll = kmid_ll
        self.khigh_ll = khigh_ll
        self.rms_tot = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[self.k_min, self.k_max],
                                  print_rms=print_rms)
        self.rms_l = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[self.k_min, kmid_ll],
                                  print_rms=print_rms)
        self.rms_m = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[kmid_ll, khigh_ll],
                                  print_rms=print_rms)
        self.rms_h = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[khigh_ll, self.k_max],
                                  print_rms=print_rms)
        self.rms_mh = do_psd_rms(psd_data=pwr_opt, delta_k=self.delta_k, k_tgt_lim=[kmid_ll, self.k_max],
                                  print_rms=print_rms)
    
    def write_psd_file(self, filename, psd_data, single_precision=True):
        '''
        Write the PSD data to file.
        I think this only applies to 2D PSDs and not the radial results.
        
        Parameters:
            filename: string
                Location and name of file to save PSD data to
            psd_data: numpy array with astropy units
                PSD data to save to file
                Good options: self.psd_cal, self.psd_norm
            single_precision: Boolean
                State whether to save data at single precision to save space,
                Otherwise will save in default state (double precision)
                Defaults to True (save to single precision)
        '''
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
    '''
    Parameters:
        data - 2D numpy array with astropy units
            Surface data in 2D array. Must have surface units (nm, um, etc)
        mask - 2D numpy array
            2D array of active surface components
        dx - float with astropy units
            spatial resolution of data, must have units (no need to match data units)
        k_side - integer
            Size of the 2D PSD. Optional.
            Default is None (will be same size as the data)
        print_update - Boolean
            Printout for MVLS status update.
            Default is False (do not print)
        write_psd - Boolean
            State whether to write the PSD to file.
            Default is False (do not save PSD to file)
        psd_name - String
            If writing PSD to file, this is name of psd.
            Default is None (no name to pass, assuming do not save file)
        psd_folder - String
            If writing PSD to file, this is the location of the psd.
            Default is None (assuming do not save file)
            
    Output:
        psd - 2D numpy array with astropy units
            MVLS PSD, calibrated by the variance of the surface.
        
        lspsd_parms - list
            List of various useful components, such as dk, kmin, kmax, var, radialFreq
    '''
    
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
    
    # calculate components
    surf_var = np.var(data[mask==1]) # has units, should be data.unit**2
    
    # create the vector array for spatial coordinates
    data_side = np.shape(data)[0]
    cen = int(data_side/2)
    if data_side % 2 == 0: # even width
        yy, xx = np.mgrid[-cen:cen, -cen:cen]
    else:
        yy, xx = np.mgrid[-cen:cen+1, -cen:cen+1]
    tnx = xx * dx.value # unitless
    tny = yy * dx.value # unitless
    
    # spatial frequency components
    # Note: k_side is not necessarily the same size as the data coming in.
    if k_side == None: # if k_side not passed, match with the size of the data.
        k_side = data.shape[0]
    dk = 1/(data_side*dx.value) # unitless
    kmin = dk/dx.unit # has units
    kmax = int(k_side/2)*dk/dx.unit # has units
    
    # load the data and apply a window to it
    surf_win = data.value * han2d((data_side, data_side)) # unitless
    
    # filter the data using the mask
    mask_filter = np.where(mask==1) # automatically vectorizes the data
    tn = np.vstack((tnx[mask_filter], tny[mask_filter]))
    ytn = surf_win[mask_filter]
    
    # build the spatial frequency coordinates
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
    
    # PSD assembly
    # the mvls psd is the raw PSD that must be scaled by the variance.
    psd_raw = ((ak**2) + (bk**2)) # unitless, but should be data.unit**2
    # normalized PSD has no units, but this will be the shape of the PSD.
    psd_norm = psd_raw / (np.sum(psd_raw) * (dk**2)) # unitless, but should be 1/dk.unit**2
    # psd is the normalized psd calibrated by the surface variance.
    psd = psd_norm * surf_var.value # unitless, but would be surf_var.unit (data.unit**2)
    psd = np.reshape(psd, (k_side, k_side)) * (data.unit*dx.unit)**2 # reshape and correctly apply units
    
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
    
    # apply all the values
    lspsd_parms = {'dk': dk/dx.unit,
                   'radialFreq': kx[0]/dx.unit, # not sure if need to keep
                   'kmin': kmin,
                   'kmax': kmax,
                   'var': surf_var}
    
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
        '''
        Use to initialize the data without a surfPSD class object.
        Parameters:
            ind_range: numpy array, integers
                Starting and end index points through the spatial and psd arrays
            k_radial: numpy array with astropy units
                Spatial frequency array
                Units: 1/m preferably
            psd_radial: numpy array with astropy units
                Radial PSD array
                Units: nm2 m2 preferably
            k_min: float with astropy units
                minimum spatial freqeuncy value for surface measurement
                units: 1/m preferably
            kmax: float with astropy units
                maximum spatial frequency value for surface measurement
                units: 1/m preferably
        '''
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
        '''
        Use this to load up the radial PSD and spatial frequency with surfPSD object.
        Parameters:
            ind_range: numpy array, integers
                Starting and end index points through the spatial and psd arrays
            psd_obj: surfPSD object
                use all the components in the surfPSD object to load up information
                Default to None (I'm not sure the benefit of this)
        '''
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
        '''
        Load in PSD parameters if predetermined outside of function
        Parameters:
            psd_parm_list: list with 5 values, separate units
                Each value in list is a different PSD parameter value
                See note above for comments on the parameters
        '''
        self.alpha = psd_parm_list[0]
        self.beta = psd_parm_list[1]
        self.L0 = psd_parm_list[2]
        self.lo = psd_parm_list[3]
        self.bsr = psd_parm_list[4]
        
    def calc_psd_parm(self, rms_sr, x0=[1.0, 1.0, 1.0, 1.0]):
        '''
        Calculate the PSD parameters through a least squares fitting function
        Parameters:
            rms_sr: float with astropy units
                Surface roughness rms value
                Units: nanometer, or at least has to match surface units.
            x0: list of floats
                "Guessing" values for non-linear least square fitting
                Could be anything, but 1.0 usually works.
        '''
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
        '''
        Calculates the PSD for full spatial frequency range (as opposed to regional)
        Parameters:
            psd_weight: float
                Weight of PSD model.
                Defaults to 1.0, but can be changed.
            k_range: numpy array with astropy units
                Spatial frequency range for PSD calculation
                Units: 1/m preferably
                Defaults to None to allow calculation based on k_spacing and k_limit
            k_spacing: float with astropy units
                Linear spacing for spatial frequency range
                Units: 1/m preferably
                Defaults to None in case k_range is passed
            k_limit: numpy array with astropy units
                Lower [0] and upper [1] bound limits for spatial frequency range
                Units: 1/m preferably
                Defaults to None in case k_range is passed
        '''
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
        '''
        Calculates the beta parameter based on alpha and rms_surf
        Parameters:
            alpha: float
                Power index value for PSD
            rms_surf: float with units
                Surface roughness rms value
                Units: nanometer (or match with surface height units)
        '''
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
    '''
    Apply the PSD model parameters to spatial frequency values
    Works for multiple PSD sets
    I'm not sure how this works, months after writing it...
    Parameters:
        x: numpy array with units(?)
            PSD parameters for fitting
        k: numpy array with units
            Spatial frequency values
    '''
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
    '''
    Fitting the modeled PSD to the PSD data
    Parameters:
        x: numpy array with units(?)
            PSD parameters for fitting
        k: numpy array with units
            Spatial frequency values
        y: numpy array with units (?)
            Radial PSD values from measured PSD
            "values to fit against" 
    '''
    pk = psd_fitter(x,k)
    return pk-y

def fit_func_ratio(x, k, y):
    '''
    Same as fit_func, but forcing a scaled value for the fitting
    Parameters:
        x: numpy array with units(?)
            PSD parameters for fitting
        k: numpy array with units
            Spatial frequency values
        y: numpy array with units (?)
            Radial PSD values from measured PSD
            "values to fit against" 
    '''
    return fit_func(x,k,y)/y

def model_beta(k_min, k_max, alpha, rms_surf):
    '''
    Models the beta value based on values
    Parameters:
        k_min: float with astropy units
            Lower limit spatial frequency value
            Units: 1/m preferably
        k_max: float with astropy units
            Upper limit spatial frequency value
            Units: 1/m preferably
        alpha: float
            PSD power index value
        rms_surf: float with astropy units
            Surface roughness rms
            Units: nanometer (or match with surface unit)
    Return:
        beta: float with astropy units
            Normalized scaling parameter value
            Has messy units
    '''
    if alpha==2:
        beta = (rms_surf**2) / (2*np.pi*np.log(k_max/k_min))
    else: # when alpha not 2
        beta = (rms_surf**2)*(alpha-2) / (2*np.pi*( (k_min**(2-alpha)) - (k_max**(2-alpha))))
    return beta # units safe

def model_bsr(k_min, k_max, rms_sr):
    '''
    Calculates the normalized surface roughness value based on parameters
    Parameters:
        k_min: float with astropy units
            Lower limit spatial frequency value
            Units: 1/m preferably
        k_max: float with astropy units
            Upper limit spatial frequency value
            Units: 1/m preferably
        rms_sr: float with astropy units
            Surface roughness rms
            Units: nanometer (or match with surface unit)
    Returns:
        bsr: float with astropy units
            Normalized surface roughness
            Matches PSD units
    '''
    return (rms_sr**2) / (np.pi * (k_max**2 - k_min**2)) # units safe

def model_full(k, psd_parm):
    '''
    Calculates a single PSD instance based on passed parameters
    Parameters:
        k: numpy array with astropy units
            Spatial frequency range
            Units: 1/m preferably
        psd_parm: list with astropy units
            List value of each PSD parameter value
            See overhead note for details
            Each PSD parameter has its own units
    '''
    # 
    alpha=psd_parm[0]
    beta=psd_parm[1]
    L0=psd_parm[2]
    lo=psd_parm[3]
    bsr=psd_parm[4]
    # verify the L0 value before passing it through
    if L0.value == 0: # need to skip out the L0 value or else it explodes
        pk = beta/((k**2)**(alpha*.5))
    else:
        if L0.unit != ((1/k).unit):# check units before calculating
            print('Changing L0 unit to match with 1/dk units')
            L0.to((1/k).unit) # match the unit with the spatial frequency
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
    '''
    We had single region PSD modeling, 
    now it's time to combine all the regions together.
    '''
    def __init__(self, mdl_set, avg_psd):
        '''
        Set up all the data parameters from the PSD model objects
        Parameters:
            mdl_set: list of model_single objects
                List of regional modeled PSD objects
            avg_psd: Not sure what sort of object
                Averaged PSD of surface measurements
        '''
        # Collect data from the average PSD object
        self.delta_k = avg_psd.delta_k
        self.npix_diam = avg_psd.npix_diam
        self.side = np.shape(avg_psd.psd_cal)[0]
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
        '''
        Apply a general fitting for the combined PSD models to
        the measured surface PSD data
        '''
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
        '''
        Overwrites the PSD parameters for each section after doing the refit
        Parameters:
            section_num: integer
                Regional section number of PSD model
            overwrite_parm: list with individual element astropy units
                List of PSD parameters to replace the original section parameters
        '''
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
        '''
        Calculates the error between the combined model PSD and measured PSD
        '''
        self.error = np.log10(self.psd_radial_sum_data.value/self.psd_radial_data.value)
        self.error_rms = np.sqrt(np.mean(np.square(self.error)))
        
    def calc_psd_rms(self, k_min=None):
        '''
        Calculates the rms of the new model PSD
        # note: changed npix_diam to side because this should work for oversampled optics too. (2020/12/15)
        '''
        if k_min==None:
            k_min = self.k_min
        self.psd_rms_sum = calc_model_rms(psd_parm=self.psd_parm, psd_weight=self.psd_weight, 
                                          side=self.side, delta_k=self.delta_k, 
                                          k_tgt_lim=[k_min, self.k_max])

# 
def plot_model2(mdl_set, model_sum, avg_psd, opt_parms, psd_range=[1e1, 1e-11],
                apply_title=False):
    '''
    Plotting assistance to show PSD model with measured data
    Parameters:
        mdl_set: list of model_single objects
            list of individual regional modeled PSD objects
        model_sum: model_combine object
            Combined model PSD
        avg_psd: surfPSD object (maybe?)
            Averaged PSD from surface measurements
        opt_parms: dictionary
            Optical parameters for surface set
        psd_range: numpy array
            Power value range to show on plot (vertical axis)
    '''
    k_radial = avg_psd.k_radial.value
    psd_radial = avg_psd.psd_radial_cal.value
    k_range_mdl = mdl_set[0].k_range.value
    
    color_list=['r', 'b', 'y', 'g', 'c']
    anno_opts = dict(xy=(0.1, .9), xycoords='axes fraction',
                     va='center', ha='center')

    plt.figure(figsize=[14,9],dpi=100, facecolor='white')
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
    ax0.set_ylim(top = psd_range[0], bottom=psd_range[1])
    ax0.set_ylabel('PSD ({0})'.format(mdl_set[0].psd_full.unit))
    ax0.legend(prop={'size':8})#,loc='center left', bbox_to_anchor=(1, 0.5))
    if apply_title == True:
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


##########################################
# WRITING DATA
'''
Set of functions for writing some data into FITS tables
'''

def psd_radial_to_fits(psd_dict, opt_name, fits_filename):
    '''
    Saves the radial PSD data into a FITS Table format.
    Parameters:
        psd_dict: dictonary
            dictionary content of different PSD things
        opt_name: string
            Name of optic
        fits_filename: string
            filename to save FITS file to
    '''
    # Assemble the table HDU for the radial PSD data
    c0 = fits.Column(name='k_radial', array=psd_dict['k_radial'].value, format='D')
    c1 = fits.Column(name='psd_radial', array=psd_dict['psd_radial'].value, format='D')
    psd_hdu = fits.BinTableHDU.from_columns([c0, c1])
    
    # make the primary header file
    hdr = fits.Header()
    hdr.set('opt_name', opt_name,
            'Name of optic')
    hdr.set('psd_unit', str(psd_dict['psd_radial'].unit),
            'PSD units')
    hdr.set('rms_unit', str(psd_dict['rms_tot'].unit),
            'rms units')
    hdr.set('var_unit', str(psd_dict['var'].unit),
            'variance units')
    hdr.set('diam_ca',psd_dict['diam_ca'].value, 
            'clear aperture diam [{0}]'.format(str(psd_dict['diam_ca'].unit)))
    hdr.set('diam_pix', psd_dict['npix_diam'],
            'number of pixels in clear aperture')
    hdr.set('oversamp', psd_dict['oversamp'],
            'FFT size array after oversampling')
    hdr.set('delta_k', psd_dict['delta_k'].value,
            'full psd spatial frequency resolution [{0}]'.format(str(psd_dict['delta_k'].unit)))
    hdr.set('kmin', psd_dict['k_min'].value,
            'minimum spatial frequency limit [{0}]'.format(str(psd_dict['k_min'].unit)))
    hdr.set('kmax', psd_dict['k_max'].value,
            'maximum spatial frequency limit [{0}]'.format(str(psd_dict['k_max'].unit)))
    hdr.set('ringsize', psd_dict['ring_width'],
            'annular size for radial PSD averaging')
    hdr.set('rms_tot', psd_dict['rms_tot'].value,
            'total rms [{0}]'.format(str(psd_dict['rms_tot'].unit)))
    hdr.set('var_tot', psd_dict['var'].value,
            'total variance used for calibrating PSD [{0}]'.format(str(psd_dict['var'].unit)))
    hdr['comment'] = 'Table 1 will be radial PSD values'
    hdr['comment'] = 'Column 0 is the radial spatial frequency value'
    hdr['comment'] = 'Column 1 is the radial PSD value for the calibrated PSD'
    empty_primary = fits.PrimaryHDU(header=hdr)
    
    # Write the FITS file
    hdul = fits.HDUList([empty_primary, psd_hdu])
    hdul.writeto(fits_filename, overwrite=True)

# Saving the combined PSD dictionary to FITS table format
def psd_model_to_fits(psd_dict, opt_name, fits_filename,
                 surf_unit=u.nm, lat_unit=u.m):
    '''
    Saves the modeled PSD data into a FITS Table format.
    Parameters:
        psd_dict: dictonary
            dictionary content of different PSD things
        opt_name: string
            Name of optic
        fits_filename: string
            filename to save FITS file to
        surf_unit: astropy quantity
            Units of surface height
            Default to nanometers
        lat_unit: astropy quantity
            Units of spatial measurement resolution
            Default to meters
    '''
    
    # get the names of the dictionary entries
    rms_name = 'psd_{0}_rms'.format(opt_name)
    weight_name = 'psd_{0}_weight'.format(opt_name)
    parm_name = 'psd_{0}'.format(opt_name)
    
    # build the values in different arrays
    psd_rms = psd_dict[rms_name].value
    psd_rms_model = psd_dict['rms_mod'].value
    psd_rms_error = psd_dict['rms_err'] # unitless
    
    # card 3 values
    k_range = psd_dict['k_range'].value
    err_range = psd_dict['err_data']
    psd_data = psd_dict['psd_data'].value
    mdl_data = psd_dict['mdl_data'].value
    
    # Card 2 values
    psd_weight = psd_dict[weight_name]
    k_start = psd_dict['k_start'].value
    k_end = psd_dict['k_end'].value
    tot_r = len(psd_dict[parm_name])
    alpha = []
    beta_val = []
    beta_mpower = []
    in_scale = []
    out_scale = []
    bsr = []

    for r in range(0, tot_r):
        psd_list = psd_dict[parm_name][r]
        alpha.append(psd_list[0])
        beta_val.append(psd_list[1].value)
        beta_unit = (psd_list[1].unit/surf_unit**2).decompose()
        beta_mpower.append(beta_unit.powers[0])
        out_scale.append(psd_list[2].value)
        in_scale.append(psd_list[3])
        bsr.append(psd_list[4].value)
        
    # Build the FITS header, which goes in the primary HDU
    hdr = fits.Header()
    hdr.set('surfname', opt_name, 'name of optic')
    hdr.set('surfunit', str(surf_unit), 'surface unit')
    hdr.set('latunit', str(lat_unit), 'lateral scale unit')
    hdr.set('rms', psd_rms, 'surface rms in surface units')
    hdr.set('rms_mod', psd_rms_model, 'model rms equivalent, in surface units')
    hdr.set('rms_err', psd_rms_error, 'error rms between measure and model, unitless')
    hdr['comment'] = 'The 1st card contains a table with all the PSD model parameters.'
    hdr['comment'] = 'Each row is the PSD parameter modeled for a specific region.'
    hdr['comment'] = 'Column 0 is alpha, unitless'
    hdr['comment'] = 'Column 1 is beta value'
    hdr['comment'] = 'Column 2 is the power value for one of beta units'
    hdr['comment'] = 'Beta units are surfunit2 latunit[col2]'
    hdr['comment'] = 'Column 3 is outer scale in surfunit units'
    hdr['comment'] = 'Column 4 is inner scale, no units'
    hdr['comment'] = 'Column 5 is the normalized surface roughness, in PSD units'
    hdr['comment'] = '(PSD units are in surfunit2 latunit2)'
    hdr['comment'] = 'Column 6 is the weight value of that PSD set into total model'
    hdr['comment'] = 'Column 7 is spatial freq. region start, units 1/latunit'
    hdr['comment'] = 'Column 8 is spatial freq. region end, units 1/latunit'
    hdr['comment'] = 'The 2nd card contains a table for the error rms.'
    hdr['comment'] = 'The error rms ranges within the measured PSD spatial freq region.'
    hdr['comment'] = 'Column 0 is the radial spatial freq value, units 1/latunit.'
    hdr['comment'] = 'Column 1 is the radial modeled PSD value, in PSD units.'
    hdr['comment'] = 'The model PSD is calculated at each spatial freq for all regions.'
    hdr['comment'] = 'Column 2 is the radial measured PSD value, in PSD units.'
    hdr['comment'] = 'Column 3 is the error rms value between measured and model.'
    hdr['comment'] = 'This is a very big mess to get the PSD model parameters, good luck.'
    empty_primary = fits.PrimaryHDU(header=hdr)
    
    # Assemble the table HDU, which is the 2nd HDU card
    c0 = fits.Column(name='alpha', array=alpha, format='D')
    c1 = fits.Column(name='beta_val', array=beta_val, format='D')
    c2 = fits.Column(name='beta_pow', array=beta_mpower, format='D')
    c3 = fits.Column(name='o_scl', array=out_scale, format='D')
    c4 = fits.Column(name='i_scl', array=in_scale, format='D')
    c5 = fits.Column(name='bsr', array=bsr, format='D')
    c6 = fits.Column(name='weight', array=psd_weight, format='D')
    c7 = fits.Column(name='k_start', array=k_start, format='D')
    c8 = fits.Column(name='k_end', array=k_end, format='D')
    table_hdu = fits.BinTableHDU.from_columns([c0, c1, c2, c3, c4, c5, c6, c7, c8])
    
    # Assemble the 3rd HDU card, which is the radial data values
    d0 = fits.Column(name='k_range', array=k_range, format='D')
    d1 = fits.Column(name='mdl_data', array=mdl_data, format='D')
    d2 = fits.Column(name='psd_data', array=psd_data, format='D')
    d3 = fits.Column(name='err_range', array=err_range, format='D')
    range_hdu = fits.BinTableHDU.from_columns([d0, d1, d2, d3])

    # write to file
    hdul = fits.HDUList([empty_primary, table_hdu, range_hdu])
    hdul.writeto(fits_filename, overwrite=True)

# Loading the combined PSD parameters from FITS file format
def load_psd_model_fits(fits_filename, 
                       surf_unit=u.nm, lat_unit=u.m):
    '''
    Loads the PSD model data from a FITS Table format.
    Parameters:
        fits_filename: string
            filename to open FITS file
        surf_unit: astropy quantity
            Units of surface height
            Default to nanometers
        lat_unit: astropy quantity
            Units of spatial measurement resolution
            Default to meters
    '''
    
    # unload the fits file
    hdul = fits.open(fits_filename)
    
    # initialize the dictionary terms
    opt_name = hdul[0].header['surfname']
    rms_name = 'psd_{0}_rms'.format(opt_name)
    weight_name = 'psd_{0}_weight'.format(opt_name)
    parm_name = 'psd_{0}'.format(opt_name)

    # unload the table and build the values
    data = hdul[1].data
    tab_parm = []
    weight_val = []
    k_start = []
    k_end = []
    for j in range(0, len(data)):
        rdata = data[j]
        tab_region = [rdata[0],
                      rdata[1]*(surf_unit**2)*(lat_unit**rdata[2]),
                      rdata[3]*lat_unit,
                      rdata[4],
                      rdata[5]*((lat_unit*surf_unit)**2)]
        tab_parm.append(tab_region)
        weight_val.append(rdata[6])
        k_start.append(rdata[7])
        k_end.append(rdata[8])

    # build the dictionary
    tdict = {parm_name: tab_parm,
             weight_name: weight_val,
             rms_name: hdul[0].header['RMS']*surf_unit,
             'k_start': k_start/lat_unit,
             'k_end': k_end/lat_unit}
    
    return tdict


###########################################
# BUILD SURFACE
'''
NOTE: This version slightly differs from poppy.wfe.PowerSpectrumWFE(), but only on how
the wfe_rms scaling occurs. The version on POPPY uses built-in utilities for defining 
coordinates and building the active aperture region used in scaling wfe_rms. The PSD shaping 
and randomizer remain the same.

The difference between this standalone code and the one in PowerSpectrumWFE is +/-0.01nm rms.
However, the scaling at each pixel may not correctly correspond with pre-solved DM maps.
The best thing to do is to use the 

If you're looking to use the Fresnel paper's pre-solved DM maps, run the get_opd() function 
in psd_wfe_poppy.py. It will require having POPPY installed.
'''
@u.quantity_input(wfe_rms=u.nm, wfe_radius=u.m, incident_angle=u.deg, pixscale=u.m)
def make_wfe_map(psd_parameters=None, psd_weight=None, pixscale=None, seed=1234,
                 samp=256, oversamp=4, wfe_rms=None, wfe_radius=None, 
                 incident_angle=0*u.deg, apply_reflection=False, map_size='crop'):
    """
    Parameters
    ----------
    psd_parameters: list (for single PSD set) or list of lists (multiple PSDs)
        List of specified PSD parameters.
        If there are multiple PSDs, then each list element is a list of specified PSD parameters.
        i.e. [ [PSD_list_0], [PSD_list_1]]
        The PSD parameters in a list are ordered as follows:
        [alpha, beta, outer_scale, inner_scale, surf_roughness]
        where:            
            alpha: float 
                The PSD index value.
            beta: astropy quantity
                The normalization constant. In units of :math: `\frac{m^{2}}{m^{\alpha-2}}`
                Numerator assumes surface units of meters
                Denominator assumes spatial frequency units are 1/m
            outer_scale: astropy quantity
                The outer scale value, where the low spatial frequency flattens. 
                Unit requirement: meters
            inner_scale: float
                Inner scale value, where the high spatial frequency flattens.
            surf_roughness: astropy quantity
                Surface roughness normalization. Should match units of PSD.
    psd_weight: iterable list of floats
        Specifies the weight muliplier to set onto each model PSD.
        If none passed, then defaults to an array of 1's for equal weight placement.
    pixscale: astropy quantity
        Spatial pixel scale resolution of the optic.
        Checks that must be in meters.
    seed : integer
        Seed for the random phase screen generator
        If none passed, defaults to 1234 seed.
    samp: integer
        Sample size for the final wfe optic. Math is done in oversampled mode. 
        If 'crop' chosen for map_size, then map_return is in samp size.
        Default to 256.
    oversamp: integer
        Ratio quantity for scaling samp to calculate PSD screen size.
        If 'full' chosen for map_size, then map_return is in oversamp*samp size.
        Default to 4.
    wfe_rms: astropy quantity
        Optional. Use this to force the wfe RMS
        If a value is passed in, this is the paraxial surface rms value (not OPD) in meters.
        If None passed, then the wfe RMS produced is what shows up in PSD calculation.
        Default to None.
    wfe_radius: astropy quantity
        Optional. If wfe_rms is passed, then the psd wfe is scaled to a beam of this radius.
        If a value is not passed in, then assumes beam diameter fills entire sample plane and uses that.
        Default to None.
    incident_angle: astropy quantity
        Adjusts the WFE based on reflected beam distortion.
        Does not distort the beam (remains circular), but will get the rms equivalent value.
        Can be passed as either degrees or radians.
        Default is 0 degrees (paraxial).
    apply_reflection: boolean
        Applies 2x scale for the OPD as needed for reflection.
        Default to False, which will only return surface.
        Set to True if the PSD model only accounts for surface and want OPD.
    map_size: string
        Choose what type of PSD WFE screen is returned.
        If 'crop', then will return the PSD WFE cropped to samp.
        If 'full', then will return the PSD WFE at the full oversampled array.
        Default to 'crop'.
        
    Returns
    -------
    map_return: numpy array with astropy quantity
        WFE map array scaled according to wfe_rms units.
    """
    
    
    # Parameter checker
    # check the incident angle units that it is not unreasonable
    if incident_angle >= 90*u.deg:
        raise ValueError("Incident angle must be less than 90 degrees, or equivalent in other units.")
        
    if wfe_rms is None: # if want to take randomized rms value, pre-set the units
        wfe_rms_unit = u.nm
    else: # verify that if wfe_rms was passed, there is also a wfe_radius component.
        wfe_rms_unit = wfe_rms.unit
        if wfe_radius is None:
            wfe_radius = pixscale * samp / 2 # assumes beam diameter fills entire sample plane
    
    # if psd_weight wasn't passed in but psd_parameters was, then default to equal weight.
    if psd_weight is None:
        psd_weight = np.ones((len(psd_parameters)))
        
    # verify the oversample isn't less than 1 (otherwise, ruins the scaling)
    if oversamp < 1:
        raise ValueError("Oversample must be no less than 1.")
    
    # verify that opd_crop is reasonable
    if map_size != 'full' and map_size != 'crop':
        raise ValueError("opd_crop needs to be either 'full' or 'crop', please try again.")
    
    # use pixelscale to calculate spatial frequency spacing
    screen_size = samp * oversamp
    dk = 1/(screen_size * pixscale) # 1/m units
    k_map = build_kmap(side=screen_size, delta_k=dk.value)
    
    # calculate the PSD
    psd_tot = np.zeros((screen_size, screen_size))
    for n in range(0, len(psd_weight)):
        # loop internal localized PSD variables
        alpha = psd_parameters[n][0]
        beta = psd_parameters[n][1]
        outer_scale = psd_parameters[n][2]
        inner_scale = psd_parameters[n][3]
        surf_roughness = psd_parameters[n][4]
        
        # unit check
        psd_units = beta.unit / ((dk.unit**2)**(alpha/2))
        assert surf_roughness.unit == psd_units, "PSD parameter units are not consistent, please re-evaluate parameters."
        surf_unit = (psd_units*(dk.unit**2))**(0.5)
        
        # initialize loop-internal PSD matrix
        psd_local = np.zeros_like(psd_tot)
        
        # calculate the PSD equation based on outer_scale presence
        if outer_scale.value == 0: # skip or else PSD explodes
            # temporary overwrite of k_map at k=0 to stop div/0 problem
            k_map[cen][cen] = 1/dk.value
            # calculate PSD as usual
            psd_denom = (k_map**2)**(alpha/2)
            # calculate the immediate PSD value
            psd_interm = (beta.value*np.exp(-((k_map*inner_scale)**2))/psd_denom)
            # overwrite PSD at k=0 to be 0 instead of infinity
            psd_interm[cen][cen] = 0
            # return k_map back to original state
            k_map[cen][cen] = 0
        else:
            psd_denom = ((outer_scale.value**(-2)) + (k_map**2))**(alpha/2) # unitless currently
            psd_interm = (beta.value*np.exp(-((k_map*inner_scale)**2))/psd_denom)
            
        # apply the surface roughness
        psd_local = psd_interm + surf_roughness.value
        
        # apply the sum with the weight of the PSD model
        psd_tot = psd_tot + (psd_weight[n] * psd_local) # should all be m2 [surf_unit]2, but unitless for all calc
        
    # set the random noise
    psd_random = np.random.RandomState()
    psd_random.seed(seed)
    rndm_noise = np.fft.fftshift(np.fft.fft2(psd_random.normal(size=(screen_size, screen_size))))
    
    psd_scaled = (np.sqrt(psd_tot/(pixscale.value**2)) * rndm_noise)
    wfe_map = ((np.fft.ifft2(np.fft.ifftshift(psd_scaled)).real*surf_unit).to(wfe_rms_unit)).value
    
    # set the rms value based on the active region of the beam
    if wfe_rms is not None:
        # build the spatial map
        r_map = build_kmap(side=screen_size, delta_k=pixscale.value)
        circ = r_map < wfe_radius.value
        active_ap = wfe_map[circ==True]
        rms_measure = np.sqrt(np.mean(np.square(active_ap))) # measured rms from declared aperture
        wfe_map *= (wfe_rms/rms_measure).value # appropriately scales entire opd
        
    # apply angle adjustment for rms
    if incident_angle.value != 0:
        wfe_map /= np.cos(incident_angle).value
        
    # Set the reflection
    if apply_reflection == True:
        wfe_map *= 2
    
    if map_size == 'crop':
        # resize the beam to the sample size from the screen size
        if oversamp > 1:
            samp_cen = int(samp/2)
            cen = int(screen_size/2)
            map_return = wfe_map[cen-samp_cen:cen+samp_cen, cen-samp_cen:cen+samp_cen]
        else: # at 1, then return the whole thing
            map_return = wfe_map
    else:
        map_return = wfe_map
    
    return map_return*wfe_rms_unit
        
# 
def write_psdwfe(wavefront, rx_opt_data, seed_val, psd_parameters, psd_weight, wfe_folder, wfe_filename,
                 apply_reflection=False):
    '''
    This function writes the PSD WFE files to drive for easy access in a .py script
    Parameters:
        wavefront: poppy wavefront object
            wavefront object created from poppy
        rx_opt_data: dictionary? list?
            Prescription information about the optical surface
        seed_val: integer
            Seed value to lock in randomizer
        psd_parameters: list (for single PSD set) or list of lists (multiple PSDs)
            List of specified PSD parameters.
            If there are multiple PSDs, then each list element is a list of specified PSD parameters.
            i.e. [ [PSD_list_0], [PSD_list_1]]
            The PSD parameters in a list are ordered as follows:
            [alpha, beta, outer_scale, inner_scale, surf_roughness]
        psd_weight: list of floats
            Weight value of each PSD model piece to apply to full PSD
            Recommended: all 1's
        wfe_folder: string
            Location to save the WFE files
        wfe_filename: sting
            Base filename of WFE files
        apply_reflection: Boolean
            Applies 2x scale for the OPD as needed for reflection.
            Default to False, which will only return surface.
            Set to True if the PSD model only accounts for surface and want OPD.
            
    '''
    opt_name = rx_opt_data['Name']
    psd_wfe_rms = rx_opt_data['Beam_rms_nm']*u.nm
    opt_angle = rx_opt_data['Incident_Angle_deg']*u.deg
    d_beam = rx_opt_data['Beam_Diameter_m']*u.m
    
    # function outputs to units of psd_wfe_rms, so need to rescale it to meters as required for POPPY
    psd_opd = psd_wfe_poppy.get_opd(psd_parameters=psd_parameters,
                                    psd_weight=psd_weight,
                                    seed=seed_val,
                                    incident_angle=opt_angle,
                                    wfe_rms=psd_wfe_rms,
                                    apply_reflection=apply_reflection,
                                    map_size='full',
                                    wave=wavefront).to(u.m)
    
    # initialize values for FITS header
    br = 1/wavefront.oversample
    npix = int(wavefront.n / wavefront.oversample)
    wavelen = wavefront.wavelength
    pixscale = wavefront._pixelscale_m
    
    # initialize the FITS header
    fhdr = fits.Header()
    fhdr.set('opt_name', opt_name, 
                'Optical element name')
    fhdr.set('opt_ind', rx_opt_data['Optical_Element_Number'], 
                'Optical element number (j) from rx csv')
    fhdr.set('npix', npix, 
                'Sample size, pre-oversample')
    fhdr.set('oversamp', br, 
                'Oversample ratio')
    fhdr.set('wavelen', wavelen.value, 
                'Wavelength for Fresnel calc [{0}]'.format(str(wavelen.unit)))
    fhdr.set('puplscal', pixscale.value, # old files may use pixscale, but puplscal required for poppy.
                'Wavefront pixscale [{0}]'.format(str(pixscale.unit)))
    fhdr.set('d_beam', d_beam.value,
                'Beam diameter [{0}]'.format(str(d_beam.unit)))
    fhdr.set('bunit', str(psd_opd.unit), # formerly opd_unit
                'Units of opd wfe')
    fhdr.set('wfe_rms', psd_wfe_rms.value,
                'Surface rms at beam diam [{0}]'.format(str(psd_wfe_rms.unit)))
    fhdr.set('i_angle', opt_angle.value,
                'Incident angle [{0}]'.format(str(opt_angle.unit)))
    fhdr.set('opd_refl', apply_reflection,
                'Boolean: OPD reflection applied')
    fhdr.set('psd_type', rx_opt_data['surf_PSD_filename'],
                'PSD model used')
    fhdr.set('seed', seed_val,
                'Randomizer seed value')

    # write the file
    fits.writeto(wfe_folder + wfe_filename + '.fits', psd_opd.value, fhdr, overwrite=True)
    
###########################################
# INTERPOLATION

def k_interp(oap_label, kval, npix_diam, norm_1Dpsd, cal_1Dpsd, k_npts):
    '''
    Interpolates the PSD data to a different spatial frequency set
    Parameters:
        oap_label: string
            Name of OAP (or optic)
        kval: Not sure what this is
            Not sure what it is for
        npix_diam: integer
            Number of pixels in diameter
        norm_1Dpsd: numpy array 
    '''
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

def build_kmap(side, delta_k):
    cen = int(side/2)
    if side%2 == 0:
        ky, kx = np.ogrid[-cen:cen, -cen:cen]
    else:
        ky, kx = np.ogrid[-cen:cen+1, -cen:cen+1]
    ky = ky*delta_k
    kx = kx*delta_k
    kmap = np.sqrt(kx**2 + ky**2)
    return kmap
    
def calc_model_rms(psd_parm, psd_weight, side, delta_k, k_tgt_lim):
    k_map = build_kmap(side = side, delta_k = delta_k)
    
    # calculate the 2D PSD
    psd_mdl = np.zeros_like(k_map.value)
    for n in range(0, len(psd_weight)):
        mdl = model_full(k=k_map, psd_parm=psd_parm[n])
        psd_mdl = psd_mdl + (mdl.value * psd_weight[n])
    psd_2D = psd_mdl * mdl.unit
    
    # with the PSD calculated, do the RMS
    psd_rms_sum = do_psd_rms(psd_data=psd_2D, delta_k=delta_k, 
                             k_tgt_lim=k_tgt_lim, print_rms=False)
    return psd_rms_sum

def do_psd_rms(psd_data, delta_k, k_tgt_lim, print_rms=False):   
    # create the grid
    side = np.shape(psd_data)[0]
    cen = int(side/2)
    if side%2 == 0:
        my, mx = np.mgrid[-cen:cen, -cen:cen]
    else:
        my, mx = np.mgrid[-cen:cen+1, -cen:cen+1]
    ky = my*delta_k
    kx = mx*delta_k
    kmin = k_tgt_lim[0]
    kmax = k_tgt_lim[1]
    
    # make the mask
    ring_mask_out = kx**2 + ky**2 <= kmax**2
    if kmin.value != 0:
        ring_mask_in = kx**2 + ky**2 <= kmin**2
        ring_mask = ring_mask_out ^ ring_mask_in #XOR statement
    else:
        ring_mask = ring_mask_out
    ring_bin = np.extract(ring_mask, psd_data) #* psd_data.unit
    
    # calculate the rms
    rms_val = np.sqrt(np.sum(ring_bin * (delta_k**2))) # should have units
    if print_rms==True:
        print('Target range - k_min: {0:.3f} and k_high: {1:.3f}'.format(kmin, kmax))
        print('RMS value: {0:.4f}'.format(rms_val))
    return rms_val


def do_psd_rms_old(psd_data, delta_k, k_tgt_lim, print_rms=False, print_kloc=False):   
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
