#########################################
# Author: Jennifer Lumbres (contact: jlumbres@optics.arizona.edu)
# Last edit: 2017/06/15
# This file is meant to be a reference for the extra functions written for MagAO-X POPPY.

#########################################
# PACKAGE IMPORT
#########################################
#load modules
import cupy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.io import fits
from  matplotlib.colors import LogNorm
import scipy.ndimage
from skimage.draw import draw # drawing tools for dark hole mask
import copy
#POPPY
import poppy
from poppy.poppy_core import PlaneType


#########################################
# FUNCTION DEFINITIONS
#########################################

def surfFITS(file_loc, optic_type, opdunit, name, apply_reflection=False, refl_angle=0*u.deg):
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
        if apply_reflection is True: # if not built-in with FITS data
            optic_fits[0].data *= 2
        if refl_angle.value != 0: # if not built-in with FITS data
            optic_fits[0].data /= np.cos(refl_angle).value
        optic_wfe = poppy.FITSOpticalElement(name = name, opd=optic_fits, opdunits = opdunit)
    else:
        optic_wfe = poppy.FITSOpticalElement(name = name, transmission=optic_fits)
    return optic_wfe


def writeOPDfile(opd_surf_data, pixelscl, fileloc, write_file=True, apply_reflection=False):
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
    if apply_reflection == True:
        opd_surf_data *= 2
    # initialize fits file
    writeOPD = fits.PrimaryHDU(data=opd_surf_data)
    writeOPD.header.set('PUPLSCAL', pixelscl)
    writeOPD.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    writeOPD.header.set('BUNIT', 'meters')
    writeOPD.header.comments['BUNIT'] = 'opd units'
    if write_file==True:
        writeOPD.writeto(fileloc, overwrite=True)
    else:
        return writeOPD


def writeTRANSfile(trans_data, pixelscl, fileloc, write_file=True):
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
    if write_file==True:
        writetrans.writeto(fileloc, overwrite=True)
    else:
        return writetrans


def makeRxCSV(csv_file, print_names=False):
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
    sys_rx=np.genfromtxt(csv_file, delimiter=',',
                         dtype="i2,U19,U10,f8,f8,f8,f8,f8,f8,f8,U90,U90,U10,U10,f8,U10,",
                         skip_header=20,names=True)
    if print_names==True:
        print('CSV file name: %s' % csv_file)
        print('The names of the headers are:')
        print(sys_rx.dtype.names)
    return sys_rx

def gen_mag_pupil(entrance_radius, samp, bump=False):
    m1_ap = poppy.CircularAperture(name='circ', radius = entrance_radius)
    secobs = poppy.AsymmetricSecondaryObscuration(secondary_radius=np.around(entrance_radius*0.29, decimals=4),
                                                  support_angle=[45,135, -45, -135],
                                                  support_width=[0.01905, 0.0381,0.01905, 0.0381],
                                                  support_offset_x=[0, 0, 0, 0],
                                                  support_offset_y= [0.34, -0.34, 0.34, -0.34])
    opticslist = [m1_ap, secobs]
    if bump is True:
        act = gen_mag_bump(entrance_radius, samp)
        opticslist.append(act)
    m1_pupil = poppy.CompoundAnalyticOptic(opticslist=opticslist, name='Mag pupil')
    return m1_pupil

def gen_mag_bump(entrance_radius, samp):
    pixelscale = 2*entrance_radius/samp
    center_loc = int(samp/2)
    if samp == 512:
        # old version based on pixels and 512 sample space
        act_radius = 17 # pixels, scaled with vAPP pupil parameters
        ac = 362 # pixel position
        ar = 123 # pixel position
        
    elif samp == 538:
        act_radius = 18 # pixels, scaled with vAPP pupil parameters
        ac = 379 # pixel position
        ar = 128 # pixel position
        
    else:
        raise Exception('samp must be 512 or 538')
    
    act = poppy.SecondaryObscuration(secondary_radius=(pixelscale.value*act_radius),
                                    n_supports=0)
    act.shift_x = (center_loc - ac) * pixelscale.value
    act.shift_y = (center_loc - ar) * pixelscale.value
    
    return act
    

def mag_pupil_mask(samp, entrance_radius, wavelength, bump=False):
    wf = poppy.poppy_core.Wavefront(npix=samp, diam=entrance_radius*2, wavelength=wavelength)
    m1_pupil = gen_mag_pupil(entrance_radius=entrance_radius, samp=samp, bump=bump)
    return m1_pupil.get_transmission(wf)
    
def csvFresnel(rx_csv, samp, oversamp, break_plane, home_folder=None, 
               psd_dict=None, seed=None, psd_reflection=True, print_rx=False, bump=False):
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
    entrance_radius=rx_csv['Radius_m'][0]*u.m 
    
    sys_build = poppy.FresnelOpticalSystem(pupil_diameter=2*entrance_radius, npix=samp, beam_ratio=oversamp)

    # Entrance Aperture
    sys_build.add_optic(poppy.CircularAperture(radius=entrance_radius))
        
    # Build MagAO-X optical system from CSV file
    for n_optic,optic in enumerate(rx_csv): # n_optic: count, optic: value

        dz = optic['Distance_m'] * u.m # Propagation distance from the previous optic (n_optic-1)
        fl = optic['Focal_Length_m'] * u.m # Focal length of the current optic (n_optic)

        # call in surface name to see if it needs to do PSD
        wfe_filename = optic['surf_PSD_filename']
        
        # if PSD file present or surface file to open
        if wfe_filename != 'none' and optic['optic_type'] != 'none': 
            if wfe_filename[0:3] == 'psd': # 'psd' notifies to calculate wfe using psd parameters
                if print_rx==True:
                    print('{0}: calculate psd wfe'.format(optic['Name']))
                psd_parm = psd_dict[wfe_filename]
                psd_weight = psd_dict[wfe_filename+'_weight']
                #psd_wfe = psd_dict[wfe_filename+'_rms']
                psd_wfe = optic['Beam_rms_nm']*u.nm
                opt_angle = optic['Incident_Angle_deg']*u.deg
                if seed is not None:
                    #psd_seed = seed[n_optic]
                    psd_seed = seed[optic['Optical_Element_Number']]
                else:
                    psd_seed = None
                optic_surface = poppy.wfe.PowerSpectrumWFE(name = optic['surf_PSD_filename']+' PSD WFE', 
                                                        psd_parameters=psd_parm, psd_weight=psd_weight, 
                                                        seed=psd_seed, apply_reflection = psd_reflection,
                                                        wfe=psd_wfe, incident_angle=opt_angle)
            else: # need to open wfe file
                if print_rx == True:
                    print('{0}: load wfe fits file ({1})'.format(optic['Name'], optic['surf_PSD_filename']))
                surf_file_loc = optic['surf_PSD_folder'] + optic['surf_PSD_filename'] + '.fits'
                if home_folder is not None:
                    surf_file_loc = home_folder+surf_file_loc
                # call surfFITS to send out surface map
                optic_surface = surfFITS(file_loc = surf_file_loc, optic_type = optic['optic_type'], 
                                         opdunit = optic['OPD_unit'], name = optic['Name']+' WFE',
                                         refl_angle = optic['Incident_Angle_deg']*u.deg)
            
            # Add generated surface map to optical system
            sys_build.add_optic(optic_surface,distance=dz)
            
            if fl != 0: # apply power if powered optic
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name'])) 
                # no distance; surface comes first
                #sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture")) 
            if optic['Type'] not in ['pupil', 'vapp'] and optic['Radius_m'] > 0: # non-powered optic n
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

        else: # if no wfe file present (DM, focal plane, testing optical surface)
            if print_rx == True:
                print('{0}: no wfe to inject'.format(optic['Name']))
                
            if optic['Type'] == 'pupil':
                sys_build.add_optic(gen_mag_pupil(entrance_radius, samp, bump=bump))
            
            elif fl !=0: # if powered optic is being tested
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name']), distance=dz)
                if optic['Radius_m'] > 0:
                    sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # for DM, flat mirrors
            elif optic['Type'] in ['mirror', 'DM']:
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                if optic['Radius_m'] > 0:
                    sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            else: # for focal plane, science plane, lyot plane
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)

        # if the most recent optic studied was the break plane, break out of loop.
        if optic['Name'] == break_plane:
            if print_rx == True:
                print('{0}: Finish building FresnelOpticalSystem'.format(optic['Name']))
            break
        
    return sys_build



def set_fresnel_parm_string(dict_def):
    wavelen = np.round(dict_def['wavelength'].to(u.nm).value).astype(int)
    br = int(1/dict_def['beam_ratio'])
    parm_name = '{0:3}_{1:1}x_{2:3}nm'.format(dict_def['npix'], br, wavelen)
    return parm_name

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

# rewritten version of BuildLyotDMSurf
def calcLyotDMSurf(lyot_phase, fresnel_parms, magK, write_fits=False):
    samp_size = fresnel_parms['npix']
    wavelength = fresnel_parms['wavelength']
    DM_ctrl_BW = fresnel_parms['tweeter_bw']
    
    # Take FT of Lyot do LPF in spatial frequency domain
    FT_lyot = np.fft.fft2(lyot_phase) # no need to shift because will return after IFFT
    
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
    
    # return surface map, NOT OPD
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
    

