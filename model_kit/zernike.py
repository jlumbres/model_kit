'''
zernike.py
This modular function file has the following goals:
* [calc_zernike_phase] Calculates Zernike phase based on circular aperture inserted
* [calc_zernike_proj] Calculates the projection weight of a Zernike on a surface. Phase only.
* [remove_ptt] Insert data, returns new data with PTT removed (and the projection weights)
'''

import numpy as np
import copy
from astropy import units as u
from scipy.special import factorial
from model_kit import datafiles as dfx

######################################

def calc_osa_index(n,m):
    '''
    Given Zernike radial degree (n) and azimuthal degree (m),
    Convert to OSA/ANSI index sequence.
    '''
    return ((n*(n+2))+m)/2


def convert_osa(j):
    '''
    Given OSA/ANSI index (j), convert to radial degree (n) and azimuthal degree (m)
    '''
    n=0
    while j>0: #iterate through to find n
        j = j-(n+1)
        if j>0:
            n = n+1
        else:
            jmod = j%(n+1)
            
    # build the m array based on n setting
    ma = np.linspace(start=-n, stop=n, num=n+1, endpoint=True)
    
    # Set m value based on position of jmod-1 (to accmodate for 0 index)
    m = int(ma[jmod-1])
    
    return (n,m)


def calc_noll_index(n, m):
    '''
    Given the Zernike radial degree (n) and azimuthal degree (m),
    Convert to Noll index sequence.
    '''
    add = 0 # default 0
    cond1 = n%4 in [0,1] and m<=0
    cond2 = n%4 in [2,3] and m>=0
    if cond1 or cond2:
        add=1
    return ((n*(n+1)/2) + np.abs(m) + add)


def convert_noll(j):
    '''
    Given Noll index (j), convert to radial degree (n) and azimuthal degree (m)
    '''
    j_odd = (j%2!=0) # if meet condition, then j is odd.
    
    n=0 #initialize for counting
    while j > 0:
        j = j-(n+1)
        if j > 0:
            n = n+1
        else:
            jmod = j%(n+1)

    # build the m array based on n
    ma = np.sort(np.abs(np.linspace(start=-n, stop=n, num=n+1, endpoint=True)))

    # calculate m value based on position of (jmod-1) and if j is even/odd
    m = int(ma[jmod-1])
    if j_odd==True: # if incoming j was odd
        m = -1*m
    
    return (n, m)

def calc_zernike_phase(j_index, aperture, j_type = 'noll'):
    '''
    Calculates Normalized Zernike function given Zernike index (j_index) based on some basis type (j_type)
    Based around aperture mask
    Code is heavily influenced from Michael Hart's OPTI 528 course
    '''
    if j_type == 'noll':
        (n,m) = convert_noll(j)
    else:
        (n,m) = convert_osa(j)
        
    ma = np.absolute(m)
    if (n<ma) or (n<0): # zernike integers
        raise Exception('Try again. n must be non-negative and n>=|m|.\nGiven: n={0}, m={1}'.format(n,m))
    
    s = (n-ma)/2 # radial polynomial upper bound
    if (s-np.floor(s) != 0):
        raise Exception('Try again. n-m must be even.\nGiven: n={0}, m={1}'.format(n,m))
    
    # Set up the grid to calculate rho and theta
    ap_diam = np.shape(aperture)[0]
    c1 = -((ap_diam-1)/2)
    c2 = ap_diam+c1-1
    x = np.linspace(c1, c2, ap_diam)
    y = np.linspace(c1, c2, ap_diam)
    xv, yv = np.meshgrid(x,y)
    yv = yv[::-1]
    rho = np.sqrt((xv**2) + (yv**2))/ap_diam*2
    theta = np.arctan2(yv,xv)

    # Calculate the radial polynomial
    R = 0
    for k in range(0,np.int(s)+1): # need the +1 to include the upper bound sum number
        mult_num = ((-1)**k) * factorial(n-k) * (rho**(n-(2*k)))
        mult_den = factorial(k) * factorial( ((n+ma)/2) - k) * factorial( ((n-ma)/2) - k)
        R = R + (mult_num/mult_den)

    # Calculate the Zernike polynomial
    if m>0: # "even" zernike
        zern = np.sqrt(2*(n+1)) * R * aperture * np.cos(ma*theta)
    elif m<0: # odd zernike
        zern = np.sqrt(2*(n+1)) * R * aperture * np.sin(ma*theta)
    else: # m==0
        zern = np.sqrt(n+1) * R * aperture
        
    # No need to tighten the matrix because passed in the aperture
    return zern * u.radian

def calc_zernike_proj(data, mask, z_coeff, index_type='noll'):
    '''
    Calculates the Zernike projection presence for a particular single index value
    Index type should be changed to 
    '''
    # check the units of data that it is in phase otherwise this doesn't work
    if data.unit != u.radian:
        raise Exception('Data units must be in phase (radians)')
    # calculate the zernike surface
    zern = calc_zernike_phase(j_index=z_coeff, aperture=mask, j_type=index_type)
    # do the dot product (dp)
    vec1d = np.product(data.shape)
    dp_num = np.dot(np.reshape(data.value, (vec1d)), np.reshape(zern.value,(vec1d)))
    dp_den = np.dot(np.reshape(zern.value, (vec1d)), np.reshape(zern.value,(vec1d)))
    return (zern, dp_num/dp_den)

def remove_zernike(data, mask, wave_num, tot_j):
    '''
    Given a surface data, aperture mask, and wave number,
    Convert to phase and calculate Zernike coefficients up to tot_j (total indices, sequential)
    Return fixed surface map and Zernike coefficients array
    '''
    surf_phase = data/wave_num
    surf_fix_phase = copy.copy(surf_phase)
    zweight = np.zeros((tot_j))
    
    for ji in range(1, tot_j+1):
        (jn, jm) = convert_noll(ji)
        z_phase, w_val = calc_zernike_proj(data=surf_phase, mask=mask, zn=jn, zm=jm)
        surf_fix_phase = surf_fix_phase - (z_phase*w_val)
        zweight[ji] = w_val
        
    # change surface back to units of OPD
    surf_fix = surf_fix_phase * wave_num
    
    # mean subtraction
    surf_mean = np.mean(surf_fix[mask==True])
    surf_fix = surf_fix - surf_mean
    
    return (surf_fix, zweight)

def remove_ptt(data, mask, wave_num):
    '''
    Shortcut method removal for only piston, tilt, tip
    '''
    rem_ptt, w_ptt = remove_zernike(data=data, mask=mask, wave_num=wave_num, tot_j=3)
    return (rem_ptt, w_ptt)


def add_zernike(data, mask, wave_num, z_weights):
    '''
    Didn't expect to do this, but here we are
    '''
    for ji in range(0, z_weights.shape[0]):
        zernike_surf = z_weights[ji] * calc_zernike_phase(j_index=ji+1, aperture=mask) * wave_num
        data = data + zernike_surf
        
    return data
    

def raw_to_surf(data_dict, write_raw=False, write_surf=False):
    tot_fm = data_dict['n_fm']
    tot_step = data_dict['n_step']
    fits_folder = data_dict['fits_folder']
    ca_resize = data_dict['ca_resize']
    diam_ca100 = data_dict['diam_ca100']
    fm_label = data_dict['fm_label']
    jmax = data_dict['n_zernike_noll']
    
    surf_cube = []
    zw = np.zeros((tot_fm, tot_step, jmax))
    for nf in range(0, tot_fm):
        fm_num = nf+1
        print('Converting flat mirror n{0} steps'.format(fm_num))
        for ns in range(0, tot_step):
            #print('Analyzing Step {0}'.format(ns))
            # call in the file
            fm_loc = 'flat_mirrors/2018_03_23/flat_{0}_n{1}_100percent_step{2}.datx'.format(fm_label,fm_num, ns)
            surf, mask, sp = dfx.open_datx(datx_file_loc=fm_loc, diam_ca100=diam_ca100)
            wavelen = sp['value'][sp['label'].index('wavelen')]

            # tighten up the matrix by removing empty rows and columns
            surf, mask = dfx.mat_tight(surf, mask)

            # apply a resize
            ca_side = np.int(np.shape(mask)[0]*ca_resize/100)
            if ca_side % 2 !=0: # check resize is even, required by PSD code
                if ca_resize < 100: 
                    ca_side += 1 # increase by 1 - better to have more
                else:
                    ca_side -= 1 # if looking at 100% CA
            ca_reduce = np.shape(mask)[0] - ca_side
            if ca_reduce > 0:
                surf, mask = dfx.mat_reduce(surf, mask, side_reduce = ca_reduce)

            # save the raw file
            if write_raw == True:
                raw_file = fits_folder+'flat_{0}_ca{1}_n{2}_step{3}_raw'.format(fm_label, ca_resize, fm_num, ns)
                dfx.write_fits(surface=surf, mask=mask, surf_parms=sp, filename=raw_file, save_mask=False)

            #Convert surface data to phase
            k_num = wavelen.to(surf.unit) / (2*np.pi*u.radian)
            surf_phase = surf / k_num

            # remove up to zernike 11, follows Noll index
            surf_fix_phase = copy.copy(surf_phase) # intitialize corrected phase
            zj_noll = np.arange(start=1, stop=jmax)
            
            for ji in zj_noll:
                (jn,jm) = convert_noll(ji)
                z_phase, w_val = calc_zernike_proj(data=surf_phase, mask=mask,zn=jn, zm=jm)
                surf_fix_phase = surf_fix_phase - (z_phase*w_val)
                zw[nf][ns][ji-1] = w_val

            # change the surface back into units of OPD
            surf_fix = surf_fix_phase * k_num
            
            # mean subtraction
            surf_mean = np.mean(surf_fix[mask==True])
            surf_fix = surf_fix - surf_mean

            # write data to a matrix
            if ns==0: # initialize first time
                data_set = np.zeros((tot_step, np.shape(mask)[0], np.shape(mask)[0])) # initialize first
            data_set[ns, :, :] = surf_fix.value

            # write all this to a FITS file
            if write_surf == True:
                fits_file = fits_folder+'flat_{0}_ca{1}_n{2}_step{3}'.format(fm_label, ca_resize, fm_num, ns)
                dfx.write_fits(surface=surf_fix, mask=mask, surf_parms=sp, filename=fits_file)
        
        surf_cube.append(data_set*surf_fix.unit)
        print('Completed flat mirror n{0} file conversion\n'.format(fm_num))
    # apply units to the data set
    data_set *= surf_fix.unit
    
    return surf_cube, zw
