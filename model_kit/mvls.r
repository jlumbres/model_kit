require(FITSio)
require(spectral)
require(tictoc)

function(home_dir, data_file, mask_file, new_file) {
    # Load the data
    #homedir <- '/home/jhen/XWCL/code/MagAOX/PSD/zygo_data/'
    #data <- readFITS(paste(homedir,'ls_test_surf.fits',sep=""))
    #mask <- readFITS(paste(homedir,'ls_dust_map.fits',sep=""))
    setwd(home_dir)
    data <- readFITS(paste(home_dir,data_file,sep="")) # data natively in microns
    mask <- readFITS(paste(home_dir,mask_file,sep=""))
    
    surf_vector <- as.vector(data$imDat)
    mask_vector <- as.vector(mask$imDat)
    
    # set up data from the fits file headers
    dx <- as.double(data$hdr[which(data$hdr=="LATRES")+1]) # dx natively in meters
    half_side <- as.double(data$hdr[which(data$hdr=="NAXIS1")+1])/2
    
    # build the spatial grid
    min_lim_xy <- -half_side * dx
    max_lim_xy <- -min_lim_xy - dx
    space <- expand.grid( x = seq(min_lim_xy, max_lim_xy, by=dx)
                          , y = seq(min_lim_xy, max_lim_xy, by=dx)
    )
    space <- space[which(mask_vector==1),] # only pick active regions
    space$val <- surf_vector[which(mask_vector==1)] # apply only the active data
    
    # begin scargling
    print("Begin scargling")
    tic('scargling')
    mvls <- spec.lomb(y=space, mode="generalized")
    toc()
    
    # write PSD data to file (whatever the units are... hrm)
    psd_vec <- mvls$PSD[2:length(mvls$PSD)] # throw away the first element, it is 0
    psd_mat <- matrix(psd_vec, nrow=sqrt(length(psd_vec)), ncol=sqrt(length(psd_vec)))
    
    dk <- mvls$f1[3] - mvls$f1[2] # they're all the same
    header <- newKwv('KEYWORD', 'VALUE', 'NOTE')
    header <- addKwv('deltak', dk, 'number, units in 1/meter', header=header)
    
    writeFITSim(psd_mat, file=new_file, header=header)
}
