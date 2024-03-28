import sys
sys.path.extend(['../'])

import torch
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path, length_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 robust_add=False, robust_drop=False, ucl_implementation=False, 
                 frame_method=None, add_rate=0.0, drop_rate=0.0):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.length_path = length_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.robust_add = robust_add
        self.robust_drop = robust_drop
        self.add_rate = add_rate
        self.drop_rate = drop_rate
        self.ucl_implementation = ucl_implementation
        self.frame_method = frame_method
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
                #self.label = self.label[0:10000]
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(
                    f, encoding='latin1')
                #self.label = self.label[0:10000]
        try:
            self.length = np.load(self.length_path)
        except:
            self.length = np.zeros((self.label.shape[0],))

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.length = self.length[0:100]
            self.sample_name = self.sample_name[0:100]


        #np.save('length_train.npy', self.length)
        #sys.exit()
        #print(self.length.shape)

        if self.robust_add is True and self.robust_drop is True:
            raise ValueError('Test either add or drop!')
        elif self.robust_add is True:
            print('start adding')
            N, C, T, V, M = self.data.shape

            if self.ucl_implementation is True:

                add_data = np.zeros((N, C, int(T* (1 + self.add_rate)), V, M))
                for i in range(len(self.data)):
                    insert_len = int((1 + self.add_rate) * int(self.length[i].item()))
                    data_numpy = self.data[i][:,:int(self.length[i].item())].transpose(3, 1, 2, 0)
                    data_rescaled = np.zeros((data_numpy.shape[0], insert_len, data_numpy.shape[2], data_numpy.shape[3]))

                    for person_id in range(data_numpy.shape[0]):
                        data_rescaled[person_id] = cv2.resize(data_numpy[person_id],
                                                            (data_numpy.shape[2],
                                                            insert_len),
                                                            cv2.INTER_LINEAR)
                    
                    tmp_add = data_rescaled.transpose(3, 1, 2, 0)
                    rest = int(T* (1 + self.add_rate)) - tmp_add.shape[1]
                    num = int(np.ceil(rest / tmp_add.shape[1]))
                    
                    pad = np.concatenate([tmp_add
                                        for _ in range(num + 1)], 1)[:, :int(T* (1 + self.add_rate))]
                    add_data[i] = pad
                    self.length[i] = int(self.length[i].item() * (1 + self.add_rate))
                #print(self.length[0])
                self.data = add_data
            else:
                print('New version')
        elif self.robust_drop is True:
            print('start dropping, drop rate is:', self.drop_rate)
            #print(self.length.shape, type(self.length))
            #np.save('length_pre_dropout_bs16.npy', self.length)

            if self.ucl_implementation is True:
                #torch.save(self.data[:32,0,:,0,0], 'data_pre_dropout_bs32.pt')
                print('drop method is ucl_implementation')

                for i in range(len(self.label)):
                    drop_index = np.random.choice(range(1, int(self.length[i].item())), size=int(
                        self.drop_rate * int(self.length[i].item())), replace=False)

                    if self.length[i].item() <= 1:
                        continue

                    # self.data[i, :, drop_index] = self.data[i, :, drop_index-1]
                    tmp_deleted = np.delete(self.data[i], drop_index, 1)[
                        :, :int(self.length[i].item() * (1 - self.drop_rate))]
                    try:
                        rest = self.data.shape[2] - tmp_deleted.shape[1]
                        num = int(np.ceil(rest / tmp_deleted.shape[1]))
                        pad = np.concatenate([tmp_deleted
                                            for _ in range(num + 1)], 1)[:, :self.data.shape[2]]
                        self.data[i] = pad
                        self.length[i] = int(self.length[i].item() * (1 - self.drop_rate))
                    except:
                        continue 

            elif self.frame_method is None or self.frame_method not in ['delete', 'repeat_previous', 'repeat_next', 'interpolate', 'expected_ucl']:
                raise ValueError('Please specify a frame method: delete, repeat_previous, repeat_next, interpolate')
            else:
                print('drop method is:', self.frame_method)
                # options: 
                # 1. Delete the frames
                # 2. repeat the previous frame
                # 3. interpolate between the previous and next frame(s)
                # 4. repeat the next available frame
                # 5. if sequence: repeat the previous and next frame, equally (i.e. meet in the middle)

                if self.frame_method == 'delete':
                    N, C, T, V, M = self.data.shape
                    data = np.zeros((N, C, T, V, M))    
                    for i in range(len(self.label)):
                        drop_indices = np.random.choice(range(1, int(self.length[i].item())), 
                                                    size=int(self.drop_rate * int(self.length[i].item())), replace=False)
                        drop_indices = np.sort(drop_indices) # sort the drop index so we don't include data we intend to delete in the next step.
                        if self.length[i].item() <= 1:
                            continue

                        trans_data = self.data[i,:,:int(self.length[i].item()),:,:]
                        new_data = self.delete_frames(trans_data, drop_indices)
                        data[i,:,:new_data.shape[1],:] = new_data
                        self.length[i] = new_data.shape[1]
                    
                    self.data = data
                
                elif self.frame_method == 'repeat_previous':
                    # repeat the previous frame
                    sort = True
                    self.repeat_previous_frame(sort)
                       
                elif self.frame_method == 'repeat_next':
                    # repeat the next frame
                    self.repeat_next_frame()
                
                elif self.frame_method == 'interpolate':
                    # interpolate between the previous and next frame
                    self.interpolate_data()
                
                elif self.frame_method == 'expected_ucl':
                    sort = False
                    self.repeat_previous_frame(sort)


    def delete_frames(self, data, drop_index):
        return np.delete(data, drop_index, 1)

    def repeat_previous_frame(self, sort=True):

        for i in range(len(self.label)):
            if self.length[i].item() <= 1:
                continue
            drop_indices = np.random.choice(range(1, int(self.length[i].item())), 
                                        size=int(self.drop_rate * int(self.length[i].item())), replace=False)
            if sort is True:
                drop_indices = np.sort(drop_indices) # sort the drop index so we don't include data we intend to delete in the next step.

            for j in range(len(drop_indices)):
                if drop_indices[j] != 1:
                    self.data[i, :, drop_indices[j]] = self.data[i, :, drop_indices[j]-1] 
                else:
                    continue
                # else if the drop index is the first frame, we can't repeat the previous frame, so don't apply this special case.  

    def repeat_next_frame(self):
        for i in range(len(self.label)):
            if self.length[i].item() <= 1:
                continue
            drop_indices = np.random.choice(range(1, int(self.length[i].item())), 
                                        size=int(self.drop_rate * int(self.length[i].item())), replace=False)
            drop_indices = np.sort(drop_indices) 
            rev_drop_indices = drop_indices[::-1] # reverse the sorted drop indices so that we're not repeating a frame we intend to delete.

            for j in range(len(rev_drop_indices)):
                if rev_drop_indices[j] != self.data[i].shape[1]-1:
                    self.data[i, :, rev_drop_indices[j]] = self.data[i, :, rev_drop_indices[j]+1]
                # else if the drop index is the last frame, we can't repeat the next frame, so don't apply this special case.

    def interpolate_data(self):  
        # Are the indices consecutive? (group them in consecutive groups)
        for i in range(len(self.label)):
            if self.length[i].item() <= 1:
                continue
            drop_indices = np.random.choice(range(1, int(self.length[i].item())), 
                                        size=int(self.drop_rate * int(self.length[i].item())), replace=False)
            drop_indices = np.sort(drop_indices)
            sub_arrays = np.split(drop_indices, np.flatnonzero(np.diff(drop_indices)!=1) + 1)
            for sub_array in sub_arrays:
                len_sub_array = len(sub_array)
                if len_sub_array > 1:
                    for j in range(len_sub_array):
                        if sub_array[len_sub_array-1]+1 == self.data.shape[2]:
                            end = self.data[i,:,sub_array[len_sub_array-1],:,:]
                        else:
                            end = self.data[i,:,sub_array[len_sub_array-1]+1,:,:]
                        start = self.data[i,:,sub_array[0]-1,:,:]
                        
                        inc = (end-start) / (len_sub_array+1)
                        self.data[i,:,sub_array[j],:,:] = inc * (j+1) + start
                else:
                    if len(sub_array) == 0:
                        continue
                    #print(len(sub_array), drop_indices.shape, self.length[i].item())
                    #print(sub_array)
                    #print(sub_arrays)
                    #print(sub_array[0])
                    #print(sub_array[0], sub_array)
                    self.data[i,:,sub_array] = (self.data[i,:,sub_array[0]-1] + self.data[i,:,sub_array[0]+1]) / 2

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        length = self.length[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, length, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


