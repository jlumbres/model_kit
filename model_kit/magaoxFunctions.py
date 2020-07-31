#########################################
# Author: Jennifer Lumbres (contact: jlumbres@optics.arizona.edu)
# Last edit: 2017/06/15
# This file is meant to be a reference for the extra functions written for MagAO-X POPPY.

#########################################
# PACKAGE IMPORT
#########################################
#load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.io import fits
from  matplotlib.colors import LogNorm
import scipy.ndimage
from skimage.draw import draw # drawing tools for dark hole mask
#POPPY
import poppy
from poppy.poppy_core import PlaneType


#########################################
# FUNCTION DEFINITIONS
#########################################

def surfFITS(file_loc, optic_type, opdunit, name):
    '''
    Initiates a FITS file to add to optical system.
    Input Parameters:
        file_loc : string
            Path location of FITS file
        optic_type : string
            Declare if the file is OPD or Transmission type ('opd' or 'trans')
        opdunit : string
            OPD units of FITS file. For some reason, BUNIT header card gives errors.
        name : string
            Descriptive name for optic. Useful for phase description.
    Output Parameters:
        optic_surf : FITSOpticalElement
            Returns FITSOpticalElement to use as surface mapping file.
    
    Sequence of Initializing:
        - Call in FITS file
        - Typecast FITS data to float type (workaround to get POPPY to accept FITS data)
        - Determine optic type to choose how to build FITSOpticalElement
        - Return FITSOpticalElement object
    '''
    optic_fits = fits.open(file_loc)
    optic_fits[0].data = np.float_(optic_fits[0].data) # typecasting for POPPY workaround
    if optic_type == 'opd':
        optic_surf = poppy.FITSOpticalElement(name = name, opd=optic_fits, opdunits = opdunit)
    else:
        optic_surf = poppy.FITSOpticalElement(name = name, transmission=optic_fits)
    return optic_surf


def writeOPDfile(opd_surf_data, pixelscl, fileloc):
    '''
    Writes OPD mask to FITS file, WILL OVERRIDE OLD FILE IF fileloc IS REUSED
    Input Parameters:
        opd_surf_data : float
            OPD surface data
        pixelscl : astropy quantity
            Pixel scale in linear astropy units, should be m/pix
        fileloc : string
            File location to save vAPP OPD mask FITS file
    Output:
        none (just does the thing)
    '''
    writeOPD = fits.PrimaryHDU(data=opd_surf_data)
    writeOPD.header.set('PUPLSCAL', pixelscl)
    writeOPD.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    writeOPD.header.set('BUNIT', 'meters')
    writeOPD.header.comments['BUNIT'] = 'opd units'
    writeOPD.writeto(fileloc, overwrite=True)


def writeTRANSfile(trans_data, pixelscl, fileloc):
    '''
    Writes transmission mask to FITS file, WILL OVERRIDE OLD FILE IF fileloc IS REUSED
    Input Parameters:
        trans_data : float
            Transmission data matrix, usually a pupil
        pixelscl : astropy quantity
            Pixel scale in linear astropy units, should be m/pix
        fileloc : string
            file location to save vAPP transmission mask FITS file
    Output:
        none (just does the thing)
    '''
    writetrans = fits.PrimaryHDU(data=trans_data)
    writetrans.header.set('PUPLSCAL', pixelscl)
    writetrans.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    writetrans.writeto(fileloc, overwrite=True)


def makeRxCSV(csv_file):
    '''
    Get the system prescription from CSV file
    FYI: This has some hardcoded numbers in it, but just follow the specs on the CSV file.
    Input parameters:
        csv_file : string
            CSV file location to open
    Output parameters:
        sys_rx : numpy array? I forget
            System prescription into a workable array format
    '''
    sys_rx=np.genfromtxt(csv_file, delimiter=',', dtype="i2,U19,U10,f8,f8,f8,U90,U90,U10,U10,f8,U10,", skip_header=15,names=True)
    print('CSV file name: %s' % csv_file)
    print('The names of the headers are:')
    print(sys_rx.dtype.names)
    return sys_rx


def csvFresnel(rx_csv, samp, oversamp, break_plane, psd_dict=None, seed=None):
    '''
    Builds FresnelOpticalSystem from a prescription CSV file passed in and using PSD WFE class.
    Input parameters:
        rx_csv : probably numpy array?
            Optical system prescription built from csv file
        samp : index
            Number of pixels resolution before zero padding
        oversamp : float
            Oversampling convention used in PROPER for how big to zero pad
        break_plane : string
            Plane to break building the MagAO-X prescription
        psd_dict : dictionary
            Contains the PSD parameters listed in dictionary format for various optics
            Not necessary if using file uploads exclusively.
        seed : iterable of intergers
            Seed for the random phase screen generator used in PSD surface building.
            Not necessary if using 
    Output:
        sys_build : poppy.FresnelOpticalSystem object 
            Complete optical system built with propagation, prescriptions included
    '''
    M1_radius=rx_csv['Radius_m'][1]*u.m # Element [1] is M1 because Element [0] is the pupil mask
    
    sys_build = poppy.FresnelOpticalSystem(pupil_diameter=2*M1_radius, npix=samp, beam_ratio=oversamp)

    # Entrance Aperture
    sys_build.add_optic(poppy.CircularAperture(radius=M1_radius))

    # Build MagAO-X optical system from CSV file
    for n_optic,optic in enumerate(rx_csv): # n_optic: count, optic: value

        dz = optic['Distance_m'] * u.m # Propagation distance from the previous optic (n_optic-1)
        fl = optic['Focal_Length_m'] * u.m # Focal length of the current optic (n_optic)

        # call in surface name to see if it needs to do PSD
        surf_filename = optic['surf_PSD_filename']
        
        if surf_filename != 'none': # if PSD file present
            if surf_filename[0:3] == 'psd': # check if surface needs to be built using PSD parameters
                psd_parm = psd_dict[surf_filename]
                psd_weight = psd_dict[surf_filename+'_weight']
                if seed != 'none':
                    psd_seed = seed[n_optic]
                else:
                    psd_seed = 'none'
                optic_surface = poppy.wfe.ModelPSDWFE(name = optic['Name']+' PSD WFE', 
                                                      psd_parameters=psd_parm, psd_weight=psd_weight, seed=psd_seed)
            else: # need to open surface file
                surf_file_loc = optic['surf_PSD_folder'] + optic['surf_PSD_filename'] + '.fits'
                # call surfFITS to send out surface map
                optic_surface = surfFITS(file_loc = surf_file_loc, optic_type = optic['optic_type'], 
                                         opdunit = optic['OPD_unit'], name = optic['Name']+' surface')
            
            # Add generated surface map to optical system
            sys_build.add_optic(optic_surface,distance=dz)
            
            if fl != 0: # powered optic with surface present
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name'])) 
                # no distance; surface comes first
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            elif optic['Type'] != 'pupil': # non-powered optic but has surface present that is NOT the pupil
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

        else: # if no surface file present (DM, focal plane, testing optical surface)
            if fl !=0: # if powered optic is being tested
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name']), distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # for DM, flat mirrors
            elif optic['Type'] == 'mirror' or optic['Type'] == 'DM':
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            else: # for focal plane, science plane, lyot plane
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)

        # if the most recent optic studied was the break plane, break out of loop.
        if optic['Name'] == break_plane: 
            #print('Finish building FresnelOpticalSystem at %s' % break_plane)
            break
        
    return sys_build


def calcFraunhofer(fresnel_parms, pupil_file, vapp_pos_file, vapp_neg_file, vapp_trans_file):
    pupil_scale = fits.open(pupil_file)[0].header['PUPLSCAL']
    pupil = surfFITS(file_loc=pupil_file, optic_type='trans', opd_unit='none',
                     name='MagAO-X Pupil (unmasked)')
    
    for ip in range(0, 3):
        mfar = poppy.OpticalSystem("Fraunhofer", oversample=np.int(1/fresnel_parms['beam_ratio']))
        mfar.add_pupil(optic=pupil)
        mfar.add_image(name='f11 fp')
        mfar.add_pupil(name='woofer DM')
        mfar.add_image(name='f16 fp')
        mfar.add_pupil(name='tweeter DM')
        mfar.add_image(name='f57 fp')
        # choose the vAPP setting
        if ip==0: # leakage term
            mfar.add_pupil(name='vAPP (none)')
        elif ip==1: # positive phase
            mfar.add_pupil(poppy.FITSOpticalElement(transmission=vapp_trans_file,
                                                    opd=vapp_pos_file,
                                                    pixelscale=pupil_scale))
        elif ip==2: # negative phase
            mfar.add_pupil(poppy.FITSOpticalElement(transmission=vapp_trans_file,
                                                    opd=vapp_neg_file,
                                                    pixelscale=pupil_scale))
        mfar.add_image(name='f69 fp')
        mfar.add_pupil(name='Lyot')
        mfar.add_image(name='f69 fp sci')
        
        # calculate the PSFs
        if ip==0: # leakage PSF
            print('Calculating Leakage PSF')
            leak_psf_far = mfar.calc_psf(wavelength=fresnel_parms['wavelength'].value)[0]
        elif ip==1: # positive phase PSF (bottom PSF)
            print('Calculating +phase PSF')
            pos_psf_far = mfar.calc_psf(wavelength=fresnel_parms['wavelength'].value)[0]
        elif ip==2: # negative phase PSF (top PSF)
            print('Calculating -phase PSF\n')
            neg_psf_far = mfar.calc_psf(wavelength=fresnel_parms['wavelength'].value)[0]
    
    # sum the PSF intensities
    tot_psf_far = pos_psf_far.data + neg_psf_far.data + (leak_psf_far.data*fresnel_parms['leak_mult'])
    
    return tot_psf_far



# Function: CropMaskLyot
# Description: Crops the Lyot phase
# NOTE: A bit obsolete now, but keeping it for old code purposes.
def CropLyot(lyot_phase_data, samp_size):
    center_pix = 0.5*(lyot_phase_data.shape[0]-1)
    shift = (samp_size/2) - 0.5 # get to center of pixel
    h_lim = np.int(center_pix+shift) # change to integer to get rid of warning
    l_lim = np.int(center_pix-shift) 
    lyot_phase = lyot_phase_data[l_lim:h_lim+1,l_lim:h_lim+1]
    return lyot_phase

# verbatim taken from numpy.pad website example
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder',0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
# independent pupil padding function
def pupilpadding(pupil_mask, fresnel_parms):
    npix = fresnel_parms['samp']
    beam_ratio = fresnel_parms['oversamp']
    
    pad_size = np.int((npix/beam_ratio)/2 - (npix/2))
    ppupil = np.pad(pupil_mask, pad_size, pad_with)
    # note - pad_with is a function directly copied from numpy.pad website example.
    
    return ppupil

# new cropping code to work with shifted and unshifted pupils
def centerCropLyotMask(lyot_data, pupil_mask, fresnel_parms, vshift=0, hshift=0):
    center_pix = 0.5*(lyot_data.shape[0]-1)
    samp_size = fresnel_parms['samp']
    beam_ratio = fresnel_parms['oversamp']
    
    shift = (samp_size/2) - 0.5 # get to center of pixel
    h_lim = np.int(center_pix+shift) # change to integer to get rid of warning
    l_lim = np.int(center_pix-shift)
    
    # pad the pupil mask
    pad_pupil = pupilpadding(pupil_mask, fresnel_parms)
    
    # shift the mask (as needed), multiply by the Lyot data, then crop.
    if vshift != 0: # only vertical shift
        pad_pupil_shift = np.roll(pad_pupil, vshift, axis=0)
        mask_data = pad_pupil_shift * lyot_data
        crop_Lyot = mask_data[l_lim+vshift:h_lim+vshift+1, l_lim:h_lim+1]
        crop_pupil = pad_pupil_shift[l_lim+vshift:h_lim+vshift+1, l_lim:h_lim+1]
    
    elif hshift != 0: # only horizontal shift
        pad_pupil_shift = np.roll(pad_pupil, hshift, axis=1)
        mask_data = pad_pupil_shift * lyot_data
        crop_Lyot = mask_data[l_lim:h_lim+1, l_lim+hshift:h_lim+hshift+1]
        crop_pupil = pad_pupil_shift[l_lim:h_lim+1, l_lim+hshift:h_lim+hshift+1]
    
    elif (vshift !=0) and (hshift != 0): # both at once  
        pad_pupil_shift0 = np.roll(pad_pupil, vshift, axis=0)
        pad_pupil_shift = np.roll(pad_pupil_shift0, hshift, axis=1)
        mask_data = pad_pupil_shift * lyot_data
        crop_Lyot = mask_data[l_lim+vshift:h_lim+vshift+1, l_lim+hshift:h_lim+hshift+1]
        crop_pupil = pad_pupil_shift[l_lim+vshift:h_lim+vshift+1, l_lim+hshift:h_lim+hshift+1]
    
    else: # neither vshift or hshift
        pad_pupil_shift = pad_pupil
        mask_data = pad_pupil_shift * lyot_data
        crop_Lyot = mask_data[l_lim:h_lim+1, l_lim:h_lim+1]
        crop_pupil = pad_pupil_shift[l_lim:h_lim+1, l_lim:h_lim+1]
        
    # return the correctly cropped Lyot and the pupil mask associated with it
    return crop_Lyot, crop_pupil


# Function: make_DHmask
# NOTE: ONLY WORKS when samp_size = 256, which is the default setting
# ALSO NOTE: I'm using hard coded numbers optimized for samp_size=256
def make_DHmask_256(samp_size):
    circ_pattern = np.zeros((samp_size, samp_size), dtype=np.uint8)
    circ_coords = draw.circle(samp_size/2, samp_size/2, radius=80)
    circ_pattern[circ_coords] = True

    rect_pattern = np.zeros((samp_size, samp_size), dtype=np.uint8)
    rect_start = (42,37)
    rect_extent = (170, 68)
    rect_coords = draw.rectangle(rect_start, extent=rect_extent, shape=rect_pattern.shape)
    rect_pattern[rect_coords] = True

    dh_mask = rect_pattern & circ_pattern # only overlap at the AND
    
    return dh_mask

# Function: BuildLyotDMSurf
# Description: Creates Tweeter DM surface map generated by Lyot plane
# Input Parameters:
#    lyot_crop          - cropped Lyot data to where it needs to be
#    lyot_phase_data    - Lyot phase data in matrix format (REMOVED)
#    samp_size          - The sampling size for the system
#    pupil_mask_data    - pupil mask data in matrix format, must be 512x512 image
#    magK               - Spatial Frequency magnitude map
#    DM_ctrl_BW         - Tweeter DM control bandwidth for LPF
#    wavelength         - wavelength tested
# Outputs:
#    lpf_lyot_mask      - 
def BuildLyotDMSurf(lyot_crop, samp_size, pupil_mask_data, magK, DM_ctrl_BW, wavelength):
    # Multiply Lyot with pupil mask
    #lyot_mask = pupil_mask_data*lyot_phase
    lyot_mask = lyot_crop*pupil_mask_data
    #lyot_mask = lyot_crop
    
    # Take FT of Lyot to get to focal plane for LPF
    FT_lyot = np.fft.fft2(lyot_mask)
    
    # LPF on FT_Lyot
    filter_lyot = np.zeros((samp_size,samp_size),dtype=np.complex128)
    for a in range (0,samp_size):
        for b in range (0,samp_size):
            #if (np.abs(kx[a][b]) < DM_ctrl_BW) and (np.abs(ky[a][b]) < DM_ctrl_BW): #square corner version
            if magK[a][b] < DM_ctrl_BW: # Curved corner version
                filter_lyot[a][b] = FT_lyot[a][b] # Keep FT value if less than DM BW
                
    # Post-LPF IFT
    lpf_lyot = np.fft.ifft2(filter_lyot)
    
    # Convert Phase to OPD on DM surface
    lpf_lyot_surf =(-1.0*wavelength.value/(2*np.pi))*np.real(lpf_lyot) # FINAL!
    
    # Multiply by pupil mask to clean up ringing
    lpf_lyot_mask = pupil_mask_data*np.real(lpf_lyot_surf)
    #lpf_lyot_mask = np.real(lpf_lyot_surf)
    
    # Write DM surface to file as OPD
    #writeOPDfile(lpf_lyot_mask, tweeter_diam.value/samp_size, lyot_DM_loc+'.fits')
    
    # return surface map as a varaible
    return lpf_lyot_mask

# rewritten version of BuildLyotDMSurf   - 
def calcLyotDMSurf(lyot_mask, fresnel_parms, magK, DM_ctrl_BW):
    samp_size = fresnel_parms['samp']
    wavelength = fresnel_parms['halpha']
    
    # Take FT of Lyot do LPF in spatial frequency domain
    FT_lyot = np.fft.fft2(lyot_mask)
    
    # LPF on FT_Lyot
    filter_lyot = np.zeros((samp_size,samp_size),dtype=np.complex128)
    for a in range (0,samp_size):
        for b in range (0,samp_size):
            #if (np.abs(kx[a][b]) < DM_ctrl_BW) and (np.abs(ky[a][b]) < DM_ctrl_BW): #square corner version
            if magK[a][b] < DM_ctrl_BW: # Curved corner version
                filter_lyot[a][b] = FT_lyot[a][b] # Keep FT value if less than DM BW
                
    # Post-LPF IFT to go back to spatial domain
    lpf_lyot = np.fft.ifft2(filter_lyot)
    
    # Convert Phase to OPD on DM surface
    lpf_lyot_surf =(-1.0*wavelength.value/(2*np.pi))*np.real(lpf_lyot) # FINAL!
    
    # return surface map as a varaible
    return lpf_lyot_surf

# Function: SpatFreqMap
# Description: Builds spatial frequency map to be used with 
# Input Parameters:
#    M1_radius  - radius of primary mirror  
#    num_pix    - side length of test region (512 is passed in for MagAO-X)
# Output:
#    magK       - spatial frequency map
def SpatFreqMap(M1_radius, num_pix):
    sample_rate = (M1_radius.value*2)/num_pix
    
    FT_freq = np.fft.fftfreq(num_pix,d=sample_rate)
    kx = np.resize(FT_freq,(FT_freq.size, FT_freq.size))
    
    # Build ky the slow way
    y_val=np.reshape(FT_freq,(FT_freq.size,1))
    ky=y_val
    for m in range (0,y_val.size-1):
        ky=np.hstack((ky,y_val))
    magK = np.sqrt(kx*kx + ky*ky)
    return magK
    
# Function: calcDHflux_List
# Description: Calculates a full list of flux values inside a region of the dark hole
# Input Parameters:
#    file_loc        - file location of PSF data to dig out dark hole
#    DH_center       - center pixel of dark hole
#    DH_side         - half side length of dark hole (a shift of sorts)
#    optics_list     - List of optics names
#    calcType        - choose between calculating flux in median or mean
# Output:
#    DH_flux_array   - list of flux inside the darh hole region, whether median value or mean
#    DH_flux_ref     - noneRemoved (reference) flux value
def calcFluxDH_List(file_loc, DH_center, DH_side, optics_list, calcType):
    DH_flux_array = []
    DH_flux_ref = 'none'
    
    for test_optic in optics_list:
        file_Data = fits.open(file_loc + test_optic + '.fits')[0].data
        DH_flux = calcDHflux(file_Data, DH_center, DH_side, calcType)
        DH_flux_array.append(DH_flux)
        
        if test_optic == 'noneRemoved':
            DH_flux_ref = DH_flux
    
    return DH_flux_array, DH_flux_ref


# Function: calcDHflux
# Description: Calculates the flux inside a region of the dark hole
# Input Parameters:
#    psf_data        - PSF data information to dig out dark hole
#    DH_center       - center pixel of dark hole
#    DH_side         - half side length of dark hole (a shift of sorts)
#    calcType        - choose between calculating flux in median or mean
# Output:
#    DH_flux         - flux inside the darh hole region, whether median value or mean
def calcDHflux(psf_data, DH_center, DH_side, calcType):
    peak_value = np.amax(np.amax(psf_data))
    contrast_img = psf_data/peak_value
    
    DHside_high = [x+DH_side for x in DH_center]
    DHside_low = [x-DH_side for x in DH_center]
    #DH_region = contrast_img[DHside_low[0]:DHside_high[0],DHside_low[1]:DHside_high[1]]
    DH_region = contrast_img[np.int(DHside_low[0]):np.int(DHside_high[0]),np.int(DHside_low[1]):np.int(DHside_high[1])]
    
    if calcType == 'mean':
        DH_flux = np.mean(DH_region)
    elif calcType == 'median':
        DH_flux = np.median(DH_region)
    
    return DH_flux
