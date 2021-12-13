import numpy as np
from astropy import units as u
from astropy.io import fits

def psf_crop(data, psf_cen, half_side):
    '''
    Crops the PSF to a specific region.
    Parameters:
        data: numpy array, nxn
            image data
        psf_cen: numpy array, (1x2)
            coordinates for PSF center
            1st is row (Y), 2nd is column (X)
        half_side: integer
            half the total side of cropped image
            Practically an axial radius.
            Must be smaller than data.shape[0]/2
    Returns:
        cropped data, sized 2*half_side x 2*half_side
        centered at psf_cen
    '''
    return data[psf_cen[0]-half_side:psf_cen[0]+half_side, 
                psf_cen[1]-half_side:psf_cen[1]+half_side]

def calc_LD_pix(fl, D_pupil, wavelen, pixscale):
    '''
    Calculates the lambda/D scale per pixel
    Parameters:
        fl: float, must have units [meters]
            focal length of the optic right before PSF
        D_pupil: float, must have units [meters]
            diameter of pupil for system
        wavelen: float, must have units [meters]
            wavelength of PSF beam
        pixscale: float, must have units [meters]
            size of pixel on detector
    Return:
        LD_scale: float, unitless
            value of lambda/D per pixel
    '''
    fnum = fl/D_pupil
    LD = (wavelen/D_pupil)*206265*u.arcsec
    platescale = 206265*u.arcsec/(fnum*D_pupil)
    LD_scale = (1/platescale)*(1/pixscale)*LD
    return LD_scale

def rms(data):
    '''
    Calculates the RMS of a specific set of data.
    Paramters:
        data: list or array
            Must be pre-solved to be only the active region
            Don't pass in an array with data outside pupil.
    Return:
        rms: float
            rms value of data
    '''
    return np.sqrt(np.sum(np.square(data))/len(data))

def calc_radial_profile(data, rmax):
    '''
    Calculates the radial profile of a PSF.
    Parameters:
        data: numpy array
            data with PSF contained.
            Note: works with only 1 PSF present in the data.
        rmax: integer
            number of pixels to extend out radially from PSF core
            Note: pending where PSF is located, this may become an issue.
    Returns:
        rp: numpy array of floats
            1D array of the annular radial distance flux from 0 to rmax-1
        cen: numpy array
            centroid position in (row,col) for data's PSF center
    '''
    # build the radial grid space
    xy = np.linspace(-rmax+1,rmax,rmax*2)
    xx, yy = np.meshgrid(xy,xy)
    r = np.sqrt(xx**2 + yy**2)
    
    # center the data to peak value and crop
    cen = np.argwhere(data==np.amax(data))[0]
    data_crop = psf_crop(data=data, psf_cen=cen,
                         half_side=rmax)
    
    # build the radial profile values
    rp = np.zeros((rmax))
    rp[0] = np.amax(data_crop) # initiate the max value at center
    for j in range(1, rmax):
        rmask_out = r<j
        rmask_in = r<(j-1)
        rmask_annular = rmask_out ^ rmask_in
        rp[j] = rms(data_crop[rmask_annular==1])
        
    return rp, cen

def calc_ee(data, data_ref, rmax):
    '''
    Calculates the encircled energy profile of a PSF.
    Parameters:
        data: numpy array
            data with PSF contained.
            Note: works with only 1 PSF present in the array.
        data_ref: numpy array
            data with reference PSF contained.
            This is used for calculating the total energy region.
            Note: works with only 1 PSF present in the array.
        rmax: integer
            number of pixels to extend out radially from PSF core
            Note: pending where data and data_ref PSF is located, 
                this may become an issue.
    Returns:
        ee: numpy array of floats
            1D array of the radial energy of data from 0 to rmax-1,
            scaled by the energy region from data_ref
        cen: numpy array
            centroid position in (row,col) for data's PSF center
        cen_ref: numpy array
            centroid position in (row,col) for data_ref's PSF center
    '''
    # build the radial grid space
    xy = np.linspace(-rmax+1,rmax,rmax*2)
    xx, yy = np.meshgrid(xy,xy)
    r = np.sqrt(xx**2 + yy**2)
    rmax_mask = r<rmax
    
    # center the data to peak value and crop
    cen = np.argwhere(data==np.amax(data))[0]
    data_crop = psf_crop(data=data, psf_cen=cent,
                         half_side=rmax)
    # center the reference data to peak value, crop, and get reference energy
    cen_ref = np.argwhere(data_ref==np.amax(data_ref))[0]
    data_ref_crop = psf_crop(data=data_ref, psf_cen=cen_ref,
                             half_side=rmax)
    tot_energy = np.sum(data_ref_crop[rmax_mask==1])
    
    # get radial encircled energy values
    ee = np.zeros((rmax))
    for j in range(0, rmax):
        rmask = r<j
        ee[j] = np.sum(data_crop[rmask==1])/tot_energy
        
    return ee, cen, cen_ref