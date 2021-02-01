from backend import readers
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import DataLoader
import hyperparams as hyp
import pickle
import os
import ipdb
st = ipdb.set_trace
import utils_improc
import utils_geom
from scipy.misc import imresize

from backend import inputs


class NpzRecordDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle):
        
        content_ = os.listdir(dataset_path)
        content = [i for i in content_ if i[-2:] == ".p"]
        records = [dataset_path + '/' + line.strip() for line in content]
        # st()
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
        # st()
        self.records = records
        self.shuffle = shuffle    

    def __getitem__(self, index):
        filename = self.records[index]
        d = pickle.load(open(filename,"rb"))
        d = dict(d)
                
        item_names = [
            'pix_T_cams_raw',
            'origin_T_camXs_raw',
            'camR_T_origin_raw',
            'rgb_camXs_raw',
            'xyz_camXs_raw',
        ]

        original_filename = filename
        filename = d['tree_seq_filename']
        # st()
        if self.shuffle:
            d,indexes = inputs.random_select_single(d, item_names, num_samples=hyp.S)
        else:
            d,indexes = inputs.non_random_select_single(d, item_names, num_samples=hyp.S)

        filename_g = "/".join([original_filename,str(indexes[0])])
        filename_e = "/".join([original_filename,str(indexes[1])])

        rgb_camXs = d['rgb_camXs_raw']
    
        d['tree_seq_filename'] = filename
        d['filename_e'] = filename_e
        d['filename_g'] = filename_g
        return d

    def __len__(self):
        return len(self.records)


def get_inputs():
    dataset_format = 'npz'
    all_set_inputs = {}
    for set_name in hyp.set_names:
        if hyp.sets_to_run[set_name]:
            data_path = hyp.demo_file_save_root_location
            shuffle = hyp.shuffles[set_name]
            if dataset_format == 'npz':
                if hyp.do_debug:
                    all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                    shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)
                else:
                    all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                    shuffle=shuffle, batch_size=hyp.B, num_workers=1, pin_memory=True, drop_last=True)
            else:
                assert False #what is the data format?
    return all_set_inputs