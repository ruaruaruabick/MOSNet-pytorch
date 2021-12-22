from scipy.stats.stats import _first
import torch
from utils import read_list
import os
import h5py
import numpy as np
import random
import torch.utils.data

class getdataset(torch.utils.data.Dataset):
    def __init__(self, config, seed, mode):

        self.config = config
        mos_list = read_list(os.path.join(config["data_dir"],'mos_list.txt'))
        random.seed(seed)
        random.shuffle(mos_list)
        self.max_timestep = self.getmax_timestep(config,seed)
        if mode == "train":
            self.filelist = mos_list[0:-(config["num_test"]+config["num_valid"])]
        elif mode == "valid":
            self.filelist = mos_list[-(config["num_test"]+config["num_valid"]):-config["num_test"]]
        elif mode == "test":
            self.filelist= mos_list[-config["num_test"]:]
    
    def read(self,file_path):
        data_file = h5py.File(file_path, 'r')
        mag_sgram = np.array(data_file['mag_sgram'][:])
        
        timestep = mag_sgram.shape[0]
        SGRAM_DIM = self.config["fft_size"] // 2 + 1
        mag_sgram = np.reshape(mag_sgram,(1, timestep, SGRAM_DIM))
        
        return {
            'mag_sgram': mag_sgram,
        }  

    def pad(self,array, reference_shape):
        
        result = np.zeros(reference_shape)
        result[:array.shape[0],:array.shape[1],:array.shape[2]] = array

        return result

    def getmax_timestep(self,config,seed):
        file_list = read_list(os.path.join(config["data_dir"],'mos_list.txt'))
        random.seed(seed)
        random.shuffle(file_list)

        filename = [file_list[x].split(',')[0].split('.')[0] for x in range(len(file_list))]
        for i in range(len(filename)):
            all_feat = self.read(os.path.join(config["bin_root"],filename[i]+'.h5'))
            sgram = all_feat['mag_sgram']
            if i == 0:
                feat = sgram
                max_timestep = feat.shape[1]
            else:
                if sgram.shape[1] > max_timestep:
                    max_timestep = sgram.shape[1]
        return max_timestep

    def __getitem__(self, index):
        # Read audio
        filename,mos = self.filelist[index].split(',')
        all_feat = self.read(os.path.join(self.config["bin_root"],filename[:-4]+'.h5'))
        sgram = all_feat['mag_sgram']
        ref_shape = [sgram.shape[0],self.max_timestep,sgram.shape[2]]
        
        sgram = self.pad(sgram,ref_shape)
        mos=np.asarray(float(mos)).reshape([1])
        frame_mos = np.array([mos*np.ones([sgram.shape[1],1])])
        return sgram, [mos,frame_mos.reshape((1,-1)).transpose(1,0)]

    def __len__(self):
        return len(self.filelist)