import numpy as np
from astropy import units as u

"""
This code has been ported over from POPPY's PowerSpectrumWFE class to run without installing POPPY.

Difference between Fresnel paper and version on STScI's github is declaration of wfe_radius:
    The Fresnel paper version interally calculates wfe_radius from beam diameter that fills sample plane.
    It is not a mandatory parameter when using wfe_rms.

    The official version (on STScI's github) has a wfe_radius value required to use wfe_rms.
    
The PSD calculation for both versions remain the same.

If you are running this code to insert into POPPY, pass in the following values:
    pixscale = pixscale at the wavefront (wavefront.pixelscale)
    samp = size of wavefront, which is the oversampled size (wavefront.shape)
    oversamp_ratio = 1, since it is already incorporated in samp
    opd_size='full'
POPPY prefers if you use the full wavefront PSD when passing in WFE as FITS.
"""

@u.quantity_input(wfe_rms=u.nm, wfe_radius=u.m, incident_angle=u.deg, pixscale=u.m)
def get_opd(psd_parameters=None, psd_weight=None, pixscale=None, seed=1234, samp=256, oversamp_ratio=4, 
            wfe_rms=None, wfe_radius=None, incident_angle=0*u.deg, apply_reflection=False,
            opd_size='crop'):
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
        Sample size for the final wfe optic. Math is done in oversampled mode, but returned in samp size.
        Default to 256.
    oversamp_ratio: integer
        Ratio quantity for scaling samp to calculate PSD screen size.
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
    opd_size: string
        Choose what type of PSD WFE screen is returned.
        If 'crop', then will return the PSD WFE cropped to samp.
        If 'full', then will return the PSD WFE at the full oversampled array.
        Default to 'crop'.
        
    Returns
    -------
    opd_crop: numpy array with astropy quantity
        OPD array sized to samp and scaled according to wfe_rms in units of meters.
    """
    
    # Parameter checker
    # check the incident angle units that it is not unreasonable
    if incident_angle >= 90*u.deg:
        raise ValueError("Incident angle must be less than 90 degrees, or equivalent in other units.")
    
    # verify that if wfe_rms was passed, there is also a wfe_radius component.
    if wfe_rms is not None and wfe_radius is None:
        wfe_radius = pixscale * samp / 2 # assumes beam diameter fills entire sample plane
    
    # if psd_weight wasn't passed in but psd_parameters was, then default to equal weight.
    if psd_weight is None:
        psd_weight = np.ones((len(psd_parameters)))
        
    # verify the oversample isn't smaller than 1 (otherwise, ruins the scaling)
    if oversamp_ratio < 1:
        raise ValueError("Oversample must be no less smaller than 1.")
    
    # verify that opd_crop is reasonable
    if opd_crop != 'full' or opd_crop != 'crop':
        raise ValueError("opd_crop needs to be either 'full' or 'crop', please try again.")
    
    # use pixelscale to calculate spatial frequency spacing
    screen_size = samp * oversamp_ratio
    dk = 1/(screen_size * pixscale) # 1/m units
    
    # build the spatial frequency map
    cen = int(screen_size/2)
    maskY, maskX = np.mgrid[-cen:cen, -cen:cen]
    ky = maskY*dk.value
    kx = maskX*dk.value
    k_map = np.sqrt(kx**2 + ky**2)
    
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
    opd = ((np.fft.ifft2(np.fft.ifftshift(psd_scaled)).real*surf_unit).to(wfe_rms.unit)).value
    
    # set the rms value based on the active region of the beam
    if wfe_rms is not None:
        # build the spatial map
        yy = maskY*pixscale
        xx = maskX*pixscale
        r_map = np.sqrt(xx**2 + yy**2)
        circ = r_map < wfe_radius
        active_ap = opd[circ==True]
        rms_measure = np.sqrt(np.mean(np.square(active_ap))) # measured rms from declared aperture
        opd *= (wfe_rms/rms_measure).value # appropriately scales entire opd
        
    # apply angle adjustment for rms
    if incident_angle.value != 0:
        opd /= np.cos(incident_angle).value
        
    # Set the reflection
    if apply_reflection == True:
        opd *= 2
    
    if opd_size == 'crop':
        # resize the beam to the sample size from the screen size
        if oversamp_ratio > 1:
            samp_cen = int(samp/2)
            opd_return = opd[cen-samp_cen:cen+samp_cen, cen-samp_cen:cen+samp_cen]
        else: # at 1, then return the whole thing
            opd_return = opd
    else:
        opd_return = opd
    
    return opd_return*wfe_rms.unit
