'''
datafiles.py
This modular function file has the following goals:
* [FILE HANDLING] Handling opening .datx and writing to .fits
* [INTERPOLATION] Code for interpolating missing surface data
* [MATRIX ADJUSTMENT] Adjusting matrices (removing rows/cols of 0's, setting even matrices)
* [SUPPORT PLOTTING] Center cropping and seeing 2 plots together
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from scipy import interpolate

# for calling in data
from astropy.io import fits
import h5py
from astropy import units as u

# drawing for the apertures
from skimage.draw import draw

# calling in other support functions
import poppy

############################################
# FILE HANDLING

def open_datx(datx_file_loc, diam_ca100 = 50*u.mm, set_surf_unit = u.micron):
    # opens a .datx file and gets the surface. Adapted from Kyle Van Gorkom.
    h5file = h5py.File(datx_file_loc) # must be .datx filetype for hdf5
    
    # Get surface and attributes
    surface_snum = list(h5file['Data']['Surface'])[0]
    surface = h5file['Data']['Surface'][surface_snum][()]
    surface_attrs = h5file['Data']['Surface'][surface_snum].attrs
    
    # Build mask from "No Data" attributes
    mask = np.ones_like(surface).astype(bool)
    mask[surface == surface_attrs['No Data']] = 0
    surface[~mask] = 0 # Apply mask (will always be true)
    
    # compile surface data information
    wavelen = surface_attrs['Wavelength'][0] * u.m
    latres = surface_attrs['X Converter'][0][2][1] * u.m/u.pix
    surfunit = surface_attrs['Unit'][0]
    scale_factor = surface_attrs['Interferometric Scale Factor'][0]
    
    # set corrections
    # correct the lateral resolution if the .datx file doesn't have any data inside it
    # diam_pix is not included in surf_parms in case of additional correction side open_datx
    if latres == 0: # this means there was no value inside
        diam_count = (np.amax(np.sum(mask,axis=0)), np.amax(np.sum(mask,axis=1)))
        diam_pix = np.amax(diam_count) * u.pix
        latres = diam_ca100.to(u.m) / diam_pix # units: m/pix
    
    # Convert the surface unit from fringe waves to microns
    if surfunit == b'Fringes':
        surf_out = surface * scale_factor * wavelen.to(set_surf_unit)
        surfunit = str(set_surf_unit)
    else:
        surf_out = surface
    
    # create and fill the exit parameter dictionary
    surf_parms = {'label': ['wavelen', 'latres', 'surfunit', 'diam_100'], 
                  'value': [wavelen, latres, surfunit, diam_ca100], 
                  'comment': ['Zygo wavelength [{0}]'.format(wavelen.unit), # meters
                              'Lateral resolution [{0}]'.format(latres.unit), # m/pix
                              'Surface units',
                              'Full optic diameter at 100% CA [{0}]'.format(diam_ca100.unit)]}

    return surf_out, mask, surf_parms


def open_fits(filename, diam_ca100 = 50*u.mm, set_surf_unit = u.micron):
    # open related fits file to get specific content out to be compatible with write_fits
    mask = fits.open(filename+'_mask.fits')[0].data
    surf = fits.open(filename+'_surf.fits')[0]
    surf_hdr = surf.header
    
    # declaring important components
    wavelen = surf_hdr['wavelen']*u.m
    latres = surf_hdr['latres']*u.m/u.pix
    if surf_hdr['surfunit'] == 'micron':
        surfunit = u.micron
    else:
        surfunit = set_surf_unit
    
    # open surface info and apply units
    surf_data = surf.data*surfunit
    
    surf_parms = {'label': ['wavelen', 'latres', 'surfunit', 'diam_100'], 
                  'value': [wavelen, latres, str(surfunit), diam_ca100], 
                  'comment': ['Zygo wavelength [{0}]'.format(wavelen.unit), # meters
                              'Lateral resolution [{0}]'.format(latres.unit), # m/pix
                              'Surface units',
                              'Full optic diameter at 100% CA [{0}]'.format(diam_ca100.unit)]}
                  
    return surf_data, mask, surf_parms
    
def write_fits(surface, mask, surf_parms, filename, save_mask=True, surf_nan=False):
    # write specific data to fits file
    header = fits.Header()
    # fill in header with parameters
    for j in range(0, len(surf_parms['label'])):
        label = surf_parms['label'][j]
        value = surf_parms['value'][j]
        
        # save out a particular number
        if label == 'latres': 
            latres = value
        elif label == 'diam_100':
            diam_100 = value
        
        # if unit is included, remove it for header
        if hasattr(value, 'unit'): 
            value = value.value
        
        comment = surf_parms['comment'][j]
        header.append((label, value, comment))
    
    # calculate data to add to header
    data_diam_pix = np.amax([np.amax(np.sum(mask,axis=0)), np.amax(np.sum(mask,axis=1))]) * u.pix
    data_diam = (latres*data_diam_pix).to(diam_100.unit)
    header['diam_ca'] = (data_diam.value, 'Data diameter at clear aperture [{0}]'.format(data_diam.unit))
    header['clear_ap'] = ((data_diam/diam_100*100).value, 'Clear aperture [percent]')
    
    # write mask file
    if save_mask==True:
        #hold_bitpix = header['bitpix']
        header['bitpix'] = -64
        fits.writeto(filename + '_mask.fits', mask.astype(int), header, overwrite=True)
        #header['bitpix'] = hold_bitpix
    
    # write surface file
    if surf_nan==True:  # if the surface to be written to FITS should be the masked nan version
        surface = sn_map(surface,mask)
        header['maskVal'] = ('nan', 'mask units')
        surf_filename = filename+'_surf_nan'
    else: # by default the surface written does not have the nan mask
        header['maskVal'] = ('zeros', 'mask units')
        surf_filename = filename+'_surf'
    if hasattr(surface, 'unit'):
        surf_val = surface.value
    else:
        surf_val = surface
    fits.writeto(surf_filename + '.fits', surf_val, header, overwrite=True)

def datx2fits(datx_file_loc, filename, diam_ca100=50*u.mm, set_surf_unit=u.micron, surf_nan=False):
    # Shortcut to write the .datx file into fits for the surface and mask
    # assumes that the data in the .datx file does not need edits
    surface, mask, surf_parms = open_datx(datx_file_loc=datx_file_loc, 
                                          diam_ca100=diam_ca100, 
                                          set_surf_unit=set_surf_unit)
                                          
    write_fits(surface, mask, surf_parms, filename, surf_nan=surf_nan)

############################################
# FIXING DATA ROUTINES

def apply_zern_arb(raw_surf, dust_mask, nt, opt_parms, write_file=False, folder_loc=None):
    #tot_fm = np.shape(dust_set)[0]
    #tot_step = np.shape(dust_set)[1]
    #zm_cor = np.zeros_like(dust_set)
    
    #for nf in range(0, tot_fm): # choose optic
    #    for ns in range(0, tot_step):
    # build the Zernikes using the arbitrary basis for mask holes
    zm = poppy.zernike.arbitrary_basis(aperture=dust_mask, nterms=nt, outside=0.0)

    # vectorize the mask
    vec1d = np.product(dust_mask.shape)
    mask_vec = np.reshape(dust_mask, (vec1d))
    mask_1d_coord = np.where(mask_vec==True)
    mask_vec_active = mask_vec[mask_1d_coord]

    # zero mean the data where active
    surf_vec_active = np.reshape(raw_surf,(vec1d))[mask_1d_coord]
    data_mean = np.mean(surf_vec_active) # only take mean of the region that matters
    data_surf = raw_surf - data_mean # zero mean
    data_surf = data_surf * dust_mask # re-calibrate the surface

    # change surface units from microns to radians
    data_surf = data_surf/(opt_parms['wavelen'].to(raw_surf.unit)/(2*np.pi*u.radian))

    # Calculate the Zernike removal
    zprj = np.zeros((nt))
    for j in range(0, nt):
        # choose zernike
        zactive = np.reshape(zm[j], (vec1d))[mask_1d_coord]

        # vectorize the data
        data_vec = np.reshape(data_surf.value, (vec1d))*data_surf.unit
        data_vec_active = data_vec[mask_1d_coord]

        # calculate the dot product value based on the active regions
        dp_num=np.dot(data_vec_active, zactive)
        dp_den=np.dot(zactive, zactive)
        zprj[j] = dp_num/dp_den

        # apply dot product removal 
        data_surf = (data_surf - (zprj[j]*zm[j]*u.rad)) * dust_mask

    # convert back to surface units
    final_surf = (data_surf * (opt_parms['wavelen'].to(raw_surf.unit)/(2*np.pi*u.radian))).value

    # set up the FITS file header
    if write_file == True:
        header = fits.Header()
        '''
        for n in range(0, len(sp['value'])):
            if hasattr(sp['value'][n], 'unit'):
                value = sp['value'][n].value
            else:
                value = sp['value'][n]
            header[sp['label'][n]] = (value, sp['comment'][n])
        '''
        header['wavelen'] = (opt_parms['wavelen'].value,
                             'Zygo wavelength [{0}]'.format(opt_parms['wavelen'].unit))
        header['latres'] = (opt_parms['latres'].value,
                            'Lateral resolution [{0}]'.format(opt_parms['latres'].unit))
        header['surfunit'] = (str(raw_surf.unit),
                              'Surface Units')
        header['diam_100'] = (opt_parms['diam_100CA'].value,
                              'Full optic diameter at 100% CA [{0}]'.format(opt_parms['diam_100CA'].unit))
        header['diam_ca'] = (opt_parms['diam_ca'].value, 
                             'Data diameter at clear aperture [{0}]'.format(opt_parms['diam_ca'].unit))
        header['clear_ap'] = (opt_parms['ca'], 
                              'Clear aperture [percent]')

        surf_name = 'flat_{0}_n{1}_{2}CA_step{3}'.format(opt_parms['label'], opt_parms['fm_num'], opt_parms['ca'], opt_parms['step_num'])
        if folder_loc is not None:
            surf_name = folder_loc+surf_name
            
        fits.writeto(surf_name+'_bigdust_mask.fits', dust_mask, header, overwrite=True)

        for zj in range(0, zprj.shape[0]):
            header['z{0}'.format(zj+1)] = (zprj[zj], 'Coeff. of Z{0} (Noll, POPPY arb. basis)'.format(zj+1))
        header['COMMENT'] = 'Surface has first {0} (Noll) Zernikes removed with dust not regarded'.format(nt)
        fits.writeto(surf_name+'_z{0}_surf.fits'.format(nt), final_surf, header, overwrite=True)

    return (final_surf * raw_surf.unit), zprj
    
    
############################################
# INTERPOLATION
# for when the data is bad on the surface

def sn_map(surface, mask):
    # produce the surface map with nans outside mask
    mask_bool = copy.copy(mask).astype(bool) # in case mask pushed in is not boolean
    surf_nan = copy.copy(surface)
    for r in range(0, np.shape(surface)[0]):
        for c in range(0,np.shape(surface)[1]):
            if mask_bool[r][c] == False:
                surf_nan[r][c] = np.nan
    return surf_nan

def interp_mask(ap_coord, mask_data):
    # Makes an interpolation mask when needing to set an active aperture region
    imask = np.ones(np.shape(mask_data))
    ind = []
    for n in range(0, np.shape(ap_coord)[1]):
        yy = ap_coord[0][n]
        xx = ap_coord[1][n]
        if mask_data[yy][xx] == 0:
            ind.append([yy, xx])
            imask[yy][xx] = 0
    return ind, imask

def fill_surface(surface, mask_data, ap_clear, ap_coords, method='cubic'):
    # fill in the missing bad pixels using cubic interpolation
    side_len = np.shape(surface)[0]
    ind, imask = interp_mask(ap_coords, mask_data)
    working_data = imask*surface*ap_clear
    data_values = working_data[imask.astype(bool)==True] # pick where the data is good
    yy,xx = np.indices(working_data.shape)
    mask_points = np.argwhere(imask==1) # indices where the data is good
    grid_z2 = interpolate.griddata(mask_points, data_values, (yy,xx), method=method)
    return grid_z2

############################################
# MATRIX ADJUSTMENT

def reduce_ca(data, mask, old_ca, new_ca):
    if old_ca < new_ca:
        raise Exception('New CA is larger than old CA, fix this')
    
    diam_new = mask.shape[0] * (new_ca/old_ca)
    
    # calculate the grid
    cen = int(mask.shape[0]/2)
    if mask.shape[0] % 2 == 0:
        yy, xx = np.ogrid[-cen:cen, -cen:cen]
    else:
        yy, xx = np.ogrid[-cen:cen+1, -cen:cen+1]
    r = np.sqrt(xx**2 + yy**2)
    ap_new = (r<=(diam_new/2))
    
    # tighten down the matrix
    data_newca, mask_newca = mat_tight(data=data*ap_new, mask=ap_new)
    
    return data_newca, mask_newca
    

def mat_tight(data, mask, print_mat=False):
    # make a "tight" matrix by removing the 0 rows and columns
    rmask = copy.copy(mask)
    rdata = copy.copy(data)
    if print_mat==True:
        print('Initial mask matrix shape:' + str(np.shape(rmask)))
        print('Initial data matrix shape:' + str(np.shape(rdata)))

    # check all sides to remove all 0 lines
    #print('Testing top row and left columns')
    #ir = 0
    #ic = 0
    top_row = np.sum(rmask[0])
    left_col = np.sum(rmask[:,0])
    while top_row == 0 or left_col==0:
        if top_row==0: # remove the row
            #print('Row {0} is all zeros'.format(ir))
            #ir+=1
            rmask = rmask[1:np.shape(rmask)[0]]
            rdata = rdata[1:np.shape(rdata)[0]]
            # calculate the new row sum
            top_row = np.sum(rmask[0])
        if left_col==0: # remove the column
            #print('Col {0} is all zeros'.format(ic))
            #ic+=1
            rmask = rmask[:,1:np.shape(rmask)[1]]
            rdata = rdata[:,1:np.shape(rdata)[1]]
            # calculate the new column sum
            left_col = np.sum(rmask[:,0])
    #print('New mask matrix shape:' + str(np.shape(rmask)))
    #print('New data matrix shape:' + str(np.shape(rdata)))

    #print('Testing bottom row and right columns')
    #jr = np.shape(rmask)[0]-1
    #jc = np.shape(rmask)[1]-1
    bot_row = np.sum(rmask[np.shape(rmask)[0]-1])
    right_col = np.sum(rmask[:,np.shape(rmask)[1]-1])
    while bot_row ==0 or right_col==0:
        if bot_row==0: # remove the row
            #print('Row {0} is all zeros'.format(jr))
            #jr-=1
            rmask = rmask[0:np.shape(rmask)[0]-1]
            rdata = rdata[0:np.shape(rdata)[0]-1]
            # calculate the new row sum
            bot_row = np.sum(rmask[np.shape(rmask)[0]-1])
        if right_col==0: # remove the column
            #print('Col {0} is all zeros'.format(jc))
            #jc-=1
            rmask = rmask[:,0:(np.shape(rmask)[1]-1)]
            rdata = rdata[:,0:(np.shape(rdata)[1]-1)]
            # calculate the new column sum
            right_col = np.sum(rmask[:,np.shape(rmask)[1]-1])
    if print_mat == True:
        print('New mask matrix shape:' + str(np.shape(rmask)))
        print('New data matrix shape:' + str(np.shape(rdata)))
    
    return (rdata, rmask)

def mat_reduce(data, mask, side_reduce):
    # a roundabout way of reducing a matrix size by padding then applying a smaller mask
    # this code can go odd-odd=even, odd-even=odd, and even-odd=even
    # but it cannot do even-even=even. I give up.
    #print('Original data and mask shape (before reduction)')
    #print('Initial matrix shape:' + str(np.shape(mask)))
    #print('Target diameter: {0}'.format(np.shape(mask)[0] - side_reduce))
    # choose size to pad based on reduction requirement
    if side_reduce % 2 == 0: 
        add_pad = 4
    else: 
        add_pad = 3
    ap_diam = np.shape(mask)[0] - side_reduce + 1 # this is some magic circle problem
    ap_radius = ap_diam/2
    #print('Aperture diameter: {0} (should be 1 more of target)'.format(ap_diam))
    #print('Padding the data by adding {0} to each side'.format(add_pad))
    pdata = np.pad(data, add_pad, pad_with, padder=0)*data.unit# removes unit status
    pmask = np.pad(mask, add_pad, pad_with, padder=0)
    # then make a circle aperture with the smaller diameter
    circ_ap = np.zeros((np.shape(pdata)[0], np.shape(pdata)[0]))
    circ_coords = draw.circle(np.shape(pdata)[0]/2, np.shape(pdata)[0]/2, radius=ap_radius)
    circ_ap[circ_coords] = True # the diameter that comes out will be (side - side_reduce)
    #print('new mask diam: {0} (should be on target)'.format(np.sum(circ_ap[int(np.shape(circ_ap)[0]/2)])))
    #print('Removing zeros from data')
    ndata, nmask = mat_tight(pdata*circ_ap, pmask*circ_ap)
    return (ndata, nmask)

# verbatim taken from numpy.pad website example
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder',0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
    
############################################
# SUPPORT PLOTTING

def doCenterCrop(optic_data,shift):
    side = np.shape(optic_data)[0]
    center = np.int(side/2)
    crop_data = optic_data[center-shift:center+shift,center-shift:center+shift]
    return crop_data

def show2plots(supertitle, data1, plot1_label, data2, plot2_label, match_colorbar=False, set_figsize=[8,8], set_dpi=150):
    # Shortcut function for 2 plot drawing
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=set_figsize, dpi=set_dpi)
    fig.suptitle(supertitle)
    
    # check the data first
    if hasattr(data1, 'unit'):
        data_set1 = data1.value
        cb1_label = str(data1.unit)
    else:
        data_set1 = data1
        cb1_label = ''
        
    if hasattr(data2, 'unit'):
        data_set2 = data2.value
        cb2_label = str(data2.unit)
    else:
        data_set2 = data2
        cb2_label = ''
        
    if match_colorbar==True:
        vmin = np.amin([np.amin(data_set1), np.amin(data_set2)])
        vmax = np.amax([np.amax(data_set1), np.amax(data_set2)])
        
    img1 = ax1.imshow(data_set1, origin='bottom')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("bottom", size="5%", pad=0.25)
    ax1.set_title(plot1_label)
    if match_colorbar == True:
        fig.clim(vmin=vmin, vmax=vmax)
    fig.colorbar(img1, cax=cax1, orientation='horizontal').set_label(cb1_label)

    img2 = ax2.imshow(data_set2, origin='bottom')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.25)
    ax2.set_title(plot2_label)
    if match_colorbar == True:
        fig.clim(vmin=vmin, vmax=vmax)
    fig.colorbar(img2, cax=cax2, orientation='horizontal').set_label(cb2_label)
    
def calc_axis(side_len, dx=None):
    cen = int(side_len/2)
    if (side_len % 2) == 0:
        side_axis = np.linspace(start=-cen, stop=cen, endpoint=False, num=side_len)
    else:
        side_axis = np.linspace(start=-cen, stop=cen, endpoint=True, num=side_len)
    if dx is not None:
        side_axis = side_axis * dx
    return side_axis

def show_image(data, pixscale, fig_title, data_unit=None, cbar_lim=None, dpi=100):
    surf_axis = calc_axis(np.shape(data)[0], pixscale)
    axis_low = np.amin(surf_axis)
    axis_high = np.amax(surf_axis)
    plt.figure(dpi=dpi)
    plt.imshow(data, origin='lower',
              extent=[axis_low.value, axis_high.value, axis_low.value, axis_high.value])
    plt.xlabel(surf_axis.unit)
    plt.ylabel(surf_axis.unit)
    plt.title(fig_title)
    if cbar_lim is not None:
        plt.clim(cbar_lim[0], cbar_lim[1])
    if hasattr(data, 'unit'):
        plt.colorbar().set_label(data.unit)
    elif data_unit is not None:
        plt.colorbar().set_label(data_unit)
    
############################################
# POTENTIALLY OBSOLETE
def adjustData(optic): # this might be obsolete?
    # Adjusts data so it becomes an even, square matrix.
    # make the matrix square, turns out I had old code I wrote in the past
    if np.shape(optic)[0] != np.shape(optic)[1]:
        optic_data = zeroPadSquare(optic)
    else:
        optic_data = optic
    
    # at this point, the data should be square, but is it an odd or even size?
    # All the code works only for even.
    if np.shape(optic_data)[0] % 2 != 0: # if odd, add a row and column of zeros.
        z_col = np.zeros((np.shape(optic_data)[0]))
        nt = np.vstack((optic_data,z_col))
        z_row = np.zeros((np.shape(nt)[0],1))
        surf_data = np.hstack((nt,z_row))
    else: # if even, then let it be
        surf_data=optic_data
    #print(np.shape(optic_data))
    
    return surf_data
