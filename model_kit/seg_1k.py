'''
seg_1k.py
Make some segments with the BMC 1K DM
Works best with 4 actuator sided segments.
'''

import numpy as np
from dataclasses import dataclass

class dm:
    
    def __init__(self, name, act_side, dmmap, cutoff=0.00001, **kwargs):
        self.name = name
        self.act_side = act_side # should be 32
        
        self.dm_grid = np.zeros((act_side, act_side))
        self.dm_com = np.zeros((act_side, act_side))
        
        # set the mask for the dmmap
        dmmap_mask = dmmap > cutoff
        # apply a banned actuator border map
        dmmap_mask[0] = 0
        dmmap_mask[:,0] = 0
        dmmap_mask[-1] = 0
        dmmap_mask[:,-1] = 0
        
        # get the row and column values for active actuators
        dm_rc_active = np.argwhere(dmmap_mask==1)
        
        # get the index values of the actuators
        tot_act = len(dm_rc_active)
        dm_ind = np.zeros((tot_act)).astype(int)
        for j in range(0, tot_act):
            dm_ind[j] = int((dm_rc_active[j][0]*self.act_side + dm_rc_active[j][1]))
            
        self.dm_rc_active = dm_rc_active
        self.dm_ind_active = dm_ind
        self.dm_mask = dmmap_mask
        
    def build_seg_dm(self, n_act_seg_side, rc_ref_cen, n_ring, seg_order):
        self.n_act_seg_side = n_act_seg_side
        self.n_ring = n_ring
        self.rc_ref_cen = rc_ref_cen
        
        # check the segment limit boundaries
        min_r = int(rc_ref_cen[0] - (n_ring*n_act_seg_side))
        min_c = int(rc_ref_cen[0] - (n_ring*n_act_seg_side))
        if min_r < 1 or min_c < 1:
            raise Exception('outermost segment reference corner out of bounds, resize seg or number rings')

        max_r = int(rc_ref_cen[0] + (n_ring*n_act_seg_side))
        max_c = int(rc_ref_cen[0] + (n_ring*n_act_seg_side))
        if max_r >= (self.act_side-1) or max_c >= (self.act_side-1):
            raise Exception('outermost segment reference corner out of bounds, resize seg or number rings')

        # calculate all the segment locations
        self.num_side_seg = int(((max_r - min_r)/n_act_seg_side)+1)
        self.tot_seg = int(self.num_side_seg**2)
        rc_ref_val = np.linspace(min_r, max_r, self.num_side_seg).astype(int)
        rc_mat_val = np.tile(rc_ref_val, (self.num_side_seg,1))
        seg_ref_raw = np.zeros((self.num_side_seg**2, 2)).astype(int)
        seg_ref_raw[:,0] = np.transpose(rc_mat_val).flatten() # row values
        seg_ref_raw[:,1] = rc_mat_val.flatten() # column values

        # now that all the segment list is made, need to reorganize it
        seg_ref_rc = np.zeros_like(seg_ref_raw)
        for j in range(0, len(seg_ref_rc)):
            seg_ref_rc[seg_order[j]] = seg_ref_raw[j]
        
        # now set a global class variable
        self.seg_ref_rc = seg_ref_rc
        
        # build each segment as a list
        seg = [] # start as an empty list
        for j in range(0, self.tot_seg):
            seg_samp = square_seg(seg_num=j, seg_side=self.n_act_seg_side,
                                  rc_ref=self.seg_ref_rc[j])
            seg.append(seg_samp)
        self.seg = seg
        
    def apply_ptt(self, ptt_mat):
        # given a ptt matrix, output a DM command matrix.
        
        # ptt matrix must be size self.num_side_seg**2 x 3 (the 3 is for ptt columns)
        if ptt_mat.shape[0] != self.tot_seg:
            raise Exception('ptt_mat has mismatching number of row segments (must match {0})'.format(self.tot_seg))
        if ptt_mat.shape[1] != 3:
            raise Exception('ptt mat has mismatching number of column segments (must be 3)')
        
        for j in range(0, self.tot_seg):
            # apply the ptt value onto the segment
            self.seg[j].apply_ptt(ptt_input=ptt_mat[j])
            
            # fill it into the dm command matrix
            for ji in range(0, self.n_act_seg_side**2):
                rc = self.seg[j].rc_act[ji]
                self.dm_com[rc[0]][rc[1]] = self.seg[j].act_com[ji]
            
        # apply a mask at the end
        self.dm_com *= self.dm_mask

@dataclass
class square_seg:
    seg_num: int
    seg_side: int
    rc_ref: list
    rc_act: list=None
    ptt: list=(0,0,0)
    map_piston: list=None
    map_tip: list=None
    map_tilt: list=None
    act_com: list=None
    
    def __post_init__(self):
        # set up all the actuators
        act_coord = np.zeros((self.seg_side, self.seg_side, 2)).astype(int)
        for jr in range(0, self.seg_side):
            for jc in range(0, self.seg_side):
                act_coord[jr][jc] = (self.rc_ref[0]+jr, self.rc_ref[1]+jc)
        self.rc_act = act_coord.reshape(self.seg_side**2, 2) # row list order
        
        # build the reference ptt
        self.map_piston = np.ones((self.seg_side, self.seg_side)).flatten()
        arr = np.linspace(-1, 1, num=self.seg_side)
        seg_tt = np.tile(arr, (self.seg_side,1))
        self.map_tip = seg_tt.flatten()
        self.map_tilt = np.flipud(np.transpose(seg_tt)).flatten()
        
        # build the initial full segment map
        self.act_com = np.zeros_like(self.map_piston)
        
    def apply_ptt(self, ptt_input):
        self.ptt = ptt_input
        self.act_com = (self.map_piston*self.ptt[0]) + (self.map_tip*self.ptt[1]) + (self.map_tilt*self.ptt[2])
