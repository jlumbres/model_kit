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
        # unload data from header
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
            self.diam_ca = (self.npix_diam * u.pix * self.latres).to(u.mm)
            
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
            
    def load_psd(self, psd_data, psd_type, var):
        if hasattr(var, 'unit'):
            self.var = var
        else:
            raise Exception('Variance needs units')
        if hasattr(psd_data, 'unit'):
            if psd_type=='norm':
                self.psd_norm=psd_data
                self.psd_cal = self.psd_norm * var 
            elif psd_type=='cal':
                self.psd_cal=psd_data
                self.psd_norm = self.psd_cal / var
            elif psd_type=='raw':
                self.psd_raw=psd_raw
            else:
                raise Exception('Type does not match cal, norm, raw; type sent: {0}'.format(psd_type))
        else:
            raise Exception('PSD needs units')
        
    def load_psd_parameters(self, oversamp, diam_ca, npix_diam, wavelen):
        self.oversamp = oversamp
        self.diam_ca = diam_ca
        self.npix_diam = npix_diam
        self.wavelen = wavelen
        self.calc_psd_parameters() # calculate other necessary parameters
    
    def calc_psd_parameters(self):
        self.k_min = 1/self.diam_ca
        self.k_max = 1/(2*self.diam_ca / self.npix_diam)
        self.delta_k = 1/(self.oversamp*self.diam_ca/self.npix_diam)
        
        # Set full radial frequency range
        samp_space = self.diam_ca / self.npix_diam
        ft_freq = np.fft.fftfreq(n=self.oversamp, d=samp_space)
        self.radialFreq = ft_freq[0:np.int(self.oversamp/2)]
        
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
        
        #optic_ovs = np.pad(hannWin, pad_side, pad_with)
        #if optic_ovs.shape[0] != self.oversamp: # I hope this doesn't break things
        #    self.oversamp = optic_ovs.shape[0]
        FT_wf = np.fft.fftshift(np.fft.fft2(optic_ovs))*self.data.unit # this comes out unitless, reapply
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
    
    def calc_psd_radial(self, ring_width):
        # shortcut version for basic code analysis
        (self.k_radial, self.psd_radial_cal) = self.do_psd_radial(ring_width=ring_width,
                                                                  psd_data = self.psd_cal)

    def do_psd_radial(self, ring_width, psd_data):
        # generic version of code if inputting a different ring width and PSD data set
        # make grid for average radial power value
        shift = np.int(self.oversamp/2)
        if self.oversamp%2 != 0:
            maskY, maskX = np.ogrid[-shift:shift+1, -shift:shift+1]
        else:
            maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
        
        # set up ring parameters
        if ring_width % 2 == 0:
            ring_width += 1 # increase by 1 to make it odd
        r_halfside = np.int((ring_width-1)/2)
        r = 1
        
        # initialize content
        mean_bin = [] # initialize empty list of mean PSD values
        k_val = [] # initialize empty list of spatial frequencies
        
        # chug along through the radial frequency values
        while((r+r_halfside)<shift): # while inside the region of interest
            ri = r - r_halfside # inner radius of ring
            if self.radialFreq[r].value < self.k_min.value: # verify that position r is at the low limit
                #print('test k-value too small, iterate to next')
                r+=1
            else:
                if ri > 0:
                    radial_mask = makeRingMask(maskY, maskX, ri, ring_width)
                    radial_bin = makeRingMaskBin(psd_data.value,radial_mask)
                    mean_bin.append(np.mean(radial_bin))
                    k_val.append(self.radialFreq[r].value)
                r+=ring_width # iterate to the next r value in the loop
                
        k_radial = k_val * self.radialFreq.unit
        psd_radial = mean_bin * psd_data.unit
        
        return (k_radial, psd_radial)
    
    def calc_rms_set(self, kmid_ll, khigh_ll, pwr_opt):
        # Calculate the RMS based on the k-parameter limits
        # all RMS units are same units as data and variance.
        self.kmid_ll = kmid_ll
        self.khigh_ll = khigh_ll
        self.rms_tot = self.calc_psd_rms(tgt_low=self.k_min, tgt_high=self.k_max,
                                     pwr_opt=pwr_opt)
        self.rms_l = self.calc_psd_rms(tgt_low=self.k_min, tgt_high=kmid_ll,
                                     pwr_opt=pwr_opt)
        self.rms_m = self.calc_psd_rms(tgt_low=kmid_ll, tgt_high=khigh_ll,
                                     pwr_opt=pwr_opt)
        self.rms_h = self.calc_psd_rms(tgt_low=khigh_ll, tgt_high=self.k_max,
                                        pwr_opt=pwr_opt)
        self.rms_mh = self.calc_psd_rms(tgt_low=kmid_ll, tgt_high=self.k_max,
                                        pwr_opt=pwr_opt)
    
    def calc_psd_rms(self, tgt_low, tgt_high, pwr_opt, print_rms=False, print_kloc = False):
        # all RMS units are same units as data and variance.
        if tgt_low > tgt_high:
            raise Exception('Spatial Frequency region not possible; tgt_low ({0:.3f}) greater than tgt_high ({1:.3f})'.format(tgt_low, tgt_high))
        # find the locations for k_low and k_high:
        (bin_low, k_low) = k_locate(self.radialFreq, tgt_low, print_change=print_kloc)
        (bin_high, k_high) = k_locate(self.radialFreq, tgt_high, print_change=print_kloc)
        ring_width = bin_high - bin_low
        
        # make a grid for the average radial power value
        shift = np.int(self.oversamp/2)
        if self.oversamp %2 != 0:
            maskY, maskX = np.ogrid[-shift:shift+1, -shift:shift+1]
        else:
            maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
        
        # make the mask
        radial_mask = makeRingMask(maskY, maskX, bin_low, ring_width)
        radial_bin = makeRingMaskBin(pwr_opt.value,radial_mask) * pwr_opt.unit
        
        # calculate the rms
        rms_val = np.sqrt(np.sum(radial_bin * (self.delta_k**2)))
        if print_rms==True:
            print('Target range - k_low: {0:.3f} and k_high: {1:.3f}'.format(k_low, k_high))
            print('RMS value: {0:.4f}'.format(rms_val))
        return rms_val
    
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
        
        # Write to FITS file
        fits.writeto(filename, write_data, hdr, overwrite=True)
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
class model:
    def __init__(self, region_num, ind_start, ind_end, k_radial, p_radial, k_min, k_max):
        self.region_num = region_num
        if hasattr(k_radial, 'unit') and hasattr(p_radial, 'unit'):
            self.k_radial = k_radial
            self.k_data = k_radial[ind_start:ind_end+1]
            self.p_data = p_radial[ind_start:ind_end+1]
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
    
    def solve_lsf(self):      
        # linearized equation from PSD in form y = mx + c
        y = np.log10(self.p_data.value)
        x = np.log10(self.k_data.value)
        # linear least square fit
        A = np.vstack([x, np.ones(len(x))]).T
        m,c = np.linalg.lstsq(A, y)[0] # y = mx + c linear equation
        self.alpha = -1*m # unitless naturally
        self.beta = 10**(c) * (self.surf_unit**2) / (self.k_data.unit**(-self.alpha+2))
    
    def extend_k(self, delta_k, k_limit):
        # Expand spatial freq range to see where the model takes us with the data
        N = np.ceil(k_limit / delta_k) # find how many values required to meet limit
        self.k_extend = np.arange(start=1, stop=N)*delta_k
        self.k_extend[0] = delta_k/2
    
    def calc_model_simple(self, k_range):
        # unit check and fix
        if self.beta.unit != (self.surf_unit**2 * self.k_data.unit**(self.alpha-2)):
            raise Exception('beta units do not match with surface units, k-space units, and alpha.')
        self.psd_simple = model_full(k=k_range, psd_parm=[self.alpha, self.beta, 0.0/self.k_data.unit,
                                                           0.0, 0.0*self.p_data.unit]) # units safe
    
    def calc_bsr(self, rms_sr):
        # unit check and fix
        if hasattr(rms_sr, 'unit'):
            if rms_sr.unit != self.surf_unit:     
                rms_sr.to(self.surf_unit)
            self.rms_sr = rms_sr
        else:
            raise Exception('surface roughness RMS needs units')
        self.bsr = model_bsr(k_min=self.k_min, k_max=self.k_max, rms_sr=self.rms_sr)
    
    def calc_model_full(self, L0, lo, k_range):
        # unit check and fix
        if hasattr(L0, 'unit'):
            if L0.unit != (self.k_radial.unit**-1): 
                L0.to(self.k_radial.unit**-1)
            self.L0 = L0
        else:
            raise Exception('L0 needs units')
        if hasattr(lo, 'unit'):
            raise Exception('lo is unitless, remove the units')
        else:
            self.lo = lo
        # verify the PSD units will match with bsr's unit before moving forward
        pmdl_unit_0 = self.beta.unit/(self.L0.unit**-self.alpha)
        pmdl_unit_1 = self.beta.unit/(self.k_radial.unit**self.alpha)
        if (pmdl_unit_0 != self.bsr.unit) or (pmdl_unit_1 != self.bsr.unit):
            raise Exception('PSD units not matching with Beta_sr units, something is wrong somewhere.')
        else: # units matching, move forward
            psd_parm=[self.alpha, self.beta, self.L0, self.lo, self.bsr]
            self.psd_full=model_full(k=k_range, psd_parm=psd_parm)
        
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

# Support functions for the model class
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
    #print('size of psd1D_data: {0}'.format(np.shape(psd1D_data)))
    psd1D_norm = np.zeros((ntot, k_npts)) # all k_val is same size
    #print('size of psd1D_norm: {0}'.format(np.shape(psd1D_norm)))
    k_out = np.zeros_like((psd1D_data))
    #print('size of k_out: {0}'.format(np.shape(k_out)))
    
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
    Code generated by Kyle Van Gorkom.
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
    Code generated by Kyle Van Gorkom.
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def zeroPadOversample(optic_data,oversamp):
    '''
    Makes zero pad based on oversampling requirements
    Input:
    optic_data  - 2D array of data
    oversamp    - oversampling multiplier
    Output:
    zp_wf       - zero padded wavefront
    '''
    n_row = np.shape(optic_data)[0]
    n_col = np.shape(optic_data)[1]
    
    if n_row != n_col: # check if a symmetric matrix is being passed in
        # zero pad data into symmetric matrix
        data = zeroPadSquare(optic_data)
        # recalibrate the number of rows and columns
        n_row = np.shape(data)[0]
        n_col = np.shape(data)[1]
    else:
        data = np.copy(optic_data)
    # Sample the matrix as some 2-factor value
    samp = getSampSide(data)
    
    # This is the oversampled side size
    side = samp * oversamp
    # NOTE: This will not work for an odd symmetric matrix! If you get an error, this is why.
    row_pad = np.int((side - n_row)/2)
    zp_wf = np.pad(data, (row_pad,row_pad), 'constant')
                
    return zp_wf

def getSampSide(optic_data):
    '''
    Calculates the sample side based on the largest side of image
    Input:
    optic_data  - 2D array of the data
    Output:
    samp        - sample side value
    '''
    # Choose the larger side first
    if np.shape(optic_data)[0] > np.shape(optic_data)[1]:
        samp_side = np.shape(optic_data)[0]
    else:
        samp_side = np.shape(optic_data)[1]
    
    # Choosing a sample size
    if samp_side < 512:
        if samp_side < 256:
            samp = 256
        else:
            samp = 512
    else:
        samp = 1024
    
    return samp

def do_psd_radial(ring_width, psd_data, dk, kmin):
    # generic version of code if inputting a different ring width and PSD data set
    # make grid for average radial power value
    if hasattr(psd_data, 'unit'):
        psd_data = psd_data.value
    side = np.shape(psd_data)[0]
    shift = int(side/2)
    radial_freq = np.linspace((-shift*dk), (shift*dk), side, endpoint=False)
    radial_freq = radial_freq[shift:side] # only pick the right side for content
    #shift = np.int(self.oversamp/2)
    if side%2 != 0:
        maskY, maskX = np.ogrid[-shift:shift+1, -shift:shift+1]
    else:
        maskY, maskX = np.ogrid[-shift:shift, -shift:shift]
        
    # set up ring parameters
    if ring_width % 2 == 0:
        ring_width += 1 # increase by 1 to make it odd
    r_halfside = np.int((ring_width-1)/2)
    r = 1

    # initialize content
    mean_bin = [] # initialize empty list of mean PSD values
    k_val = [] # initialize empty list of spatial frequencies
    
    # chug along through the radial frequency values
    while((r+r_halfside)<shift): # while inside the region of interest
        ri = r - r_halfside # inner radius of ring
        if radial_freq[r].value <= kmin.value: # verify that position r is at the low limit
            #print('test k-value too small, iterate to next')
            r+=1
        else:
            if ri > 0:
                radial_mask = psd.makeRingMask(maskY, maskX, ri, ring_width)
                radial_bin = psd.makeRingMaskBin(psd_data,radial_mask)
                mean_bin.append(np.mean(radial_bin))
                k_val.append(radial_freq[r].value)
            r+=ring_width # iterate to the next r value in the loop

    k_radial = k_val * radial_freq.unit
    psd_radial = mean_bin #* psd_data.unit

    return (k_radial, psd_radial)


def makeRingMask(y,x,inner_r,r_width):
    '''
    Makes radial median mask... that looks like a ring.
    Input:
    y        - meshgrid vertical values (pixel count units)
    x        - meshgrid horizontal values (pixel count units)
    inner_r  - inner radial value (pixel count units)
    dr       - ring thickness (pixel count units)
    Output:
    ringMask - ring mask (boolean type)
    '''
    inside_mask = x**2+y**2 <= inner_r**2
    outside_mask = x**2+y**2 <= (inner_r+r_width)**2
    ringMask = outside_mask ^ inside_mask # use xor, such that so long as one is true then it will make ring.
    return ringMask
    
def makeRingMaskBin(power_data, ringMask):
    '''
    Returns bin values of the ring mask
    Input:
    data        - wavefront power data, must be square matrix
    ringmask    - ring mask, must be same size as data and boolean type
    Output:
    ringmaskbin - vector of values passed through mask
    '''
    ringMaskBin = np.extract(ringMask, power_data)
    ringbin = ringMaskBin[~np.isnan(ringMaskBin)]
    return ringbin

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
