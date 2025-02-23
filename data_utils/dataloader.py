import os
import numpy as np
from torch.utils.data import Dataset
import nrrd
    
class Molar3D(Dataset):
    def __init__(self, transform=None, phase='train', parent_path=None, data_type="full"):
        
        self.data_files = []
        self.label_files = []
        self.spacing = []
        self.filename = []

        cur_path = os.path.join(parent_path, str(phase))
        for file_name in os.listdir(cur_path):
            if file_name.endswith('_volume.nrrd'):
                cur_file_abbr = file_name.split("_volume")[0]
                self.filename.append(cur_file_abbr+"_label.npy")
                
                if data_type == "full":
                    _label = np.load(os.path.join(cur_path, cur_file_abbr+"_label.npy"))
                    # if np.any(np.sum(_label,1)<0):
                    #     continue
                if data_type == "mini":
                    _label = np.load(os.path.join(cur_path, cur_file_abbr+"_label.npy"))
                    # if np.all(np.sum(_label,1)>0):
                    #     continue

                self.data_files.append(os.path.join(cur_path, cur_file_abbr+"_volume.nrrd"))
                self.label_files.append(os.path.join(cur_path, cur_file_abbr+"_label.npy"))
                self.spacing.append(os.path.join(cur_path, cur_file_abbr+"_spacing.npy"))
        
        self.transform = transform
        print('the data length is %d, for %s' % (len(self.data_files), phase))

    def __len__(self):
        L = len(self.data_files)
        return L

    def __getitem__(self, index):
        _img, _ = nrrd.read(self.data_files[index])
        _landmark = np.load(self.label_files[index])
        _spacing = np.load(self.spacing[index])
        _filename = self.filename[index]
        sample = {'image': _img, 'landmarks': _landmark, 'spacing':_spacing,'filename':_filename }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
        
    def __str__(self):
        pass      
    
