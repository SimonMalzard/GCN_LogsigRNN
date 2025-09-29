import sys
sys.path.extend(['../'])

import torch
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from feeders import tools

import scipy as sp
from scipy.spatial.transform import Rotation as R, Slerp

class Feeder(Dataset):
    def __init__(self, data_path, label_path, length_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 robust_add=False, robust_drop=False, ucl_implementation=False, 
                 frame_method=None, add_rate=0.0, drop_rate=0.0, structured_degradation=False, 
                 structured_degradation_type=None, structured_res=1, FPS=30, chunks=None,
                 mitigation=False, spatial_deg=False):
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
        :param robust_add: If true, add frames to the sequence
        :param robust_drop: If true, drop frames from the sequence
        :param add_rate: when robust_add is true, the rate of adding frames
        :param drop_rate: when robust_drop is true, the rate of dropping frames
        :param ucl_implementation: If true, use the UCL implementation of frame adding/dropping (from the unforked LogSigRNN Repo - we've established this method is buggy)
        :param frame_method: Method for frame dropping frames ('delete', 'repeat_previous', 'repeat_next', 'interpolate', 'expected_ucl')
        :param structured_degredation: If true, perform structured frame dropping, e.g. decimation or chunk dropping
        :param structured_degredation_type: Type of structured degradation ('reduced_resolution', 'frame_rate')
        :param structured_res: Resolution for decimation (e.g., only keep every n frames for 'reduced_resolution')
        :param FPS: target FPS rate for 'frame_rate' type structured degradation
        :param chunks: Number of chunks for 'chunk' type structured degradation (currently not implemented, always 1 chunk)
        :param mitigation: If true, apply mitigation strategies for structured degradation
        :param spatial_deg: If true, apply spatial degradation of the right hand instead of temporal degradation using SLERP interpolation
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
        self.robust_add = robust_add
        self.robust_drop = robust_drop
        self.add_rate = add_rate
        self.drop_rate = drop_rate
        self.ucl_implementation = ucl_implementation
        self.frame_method = frame_method
        self.structured_degradation = structured_degradation
        self.structured_degradation_type = structured_degradation_type
        self.structured_res = structured_res
        self.FPS = FPS
        self.chunks = chunks
        self.mitigation = mitigation
        self.spatial_deg = spatial_deg

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
            self.data = np.load(self.data_path, mmap_mode='r+')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.length = self.length[0:100]
            self.sample_name = self.sample_name[0:100]

        self.raw_data = self.data.copy()
        #np.save('length_train.npy', self.length)
        #sys.exit()
        #print(self.length.shape)

        if self.robust_add is True and self.robust_drop is True:
            raise ValueError('Test either add or drop!')
        if (self.robust_add is True and self.structured_degradation is True) or (self.robust_drop is True and self.structured_degradation is True):
            raise ValueError('Test either sturctured degredation or robustness tests, not both at the same time!')
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
            #np.save('draw_skeleton_data/action_mat_pre_dropout_dr_' + str(self.drop_rate) + '.npy', self.data[:32])

            if self.ucl_implementation is True:
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
                #np.save('draw_skeleton_data/action_mat_pre_dropout_dr_' + str(self.drop_rate) + '.npy', self.data[:32])
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
                    self.repeat_previous_frame(sort) ## This not implemented, defaulted to repeat_previous. 
        elif self.structured_degradation is True:
            print('start structured degradation')
            if self.structured_degradation_type == 'reduced_resolution':
                print('Degradation type:', self.structured_degradation_type)
                print('Keeping every', self.structured_res, 'frames')

                self.apply_reduced_resolution(self.data, self.length)

            elif self.structured_degradation_type == 'frame_rate':

                if self.FPS == 30:
                    sys.exit('Frame rate cannot be set to default frame rate (30 FPS)')
                    
                self.reduce_frame_rate(self.data, self.length)

        
        if self.spatial_deg is True:
            print('Applying spatial degredation')
            print('FPS:', self.FPS)
            # Apply spatial degradation
            N, C, T, V, M = self.data.shape
            data = np.zeros((N, C, T, V, M))
            for i in tqdm(range(len(self.label))):
                if self.length[i].item() <= 1:
                    continue
                no_of_frames = int(self.length[i])
                skel_data_right_hand = self.data[i, :, :no_of_frames, [9, 10, 11, 23, 24], 0].copy() # C, T, V
                skel_data_right_hand = self.apply_spatial_degradation(skel_data_right_hand, no_of_frames) # V, C, T
                #skel_data_right_hand = skel_data_right_hand.transpose(2, 0, 1)
                self.data[i, :, :no_of_frames, [9, 10, 11, 23, 24], 0] = skel_data_right_hand 


    def apply_spatial_degradation(self, skel_data_right_hand, no_of_frames):

        #print(skel_data_right_hand.shape)

        joint_elbow = skel_data_right_hand[0]
        joint_wrist = skel_data_right_hand[1]
        joint_hand = skel_data_right_hand[2]
        joint_tip_of_hand = skel_data_right_hand[3]
        joint_thumb = skel_data_right_hand[4]

        FPS_drop = self.FPS / 30
        chunk_length = int(no_of_frames-no_of_frames*FPS_drop)
        #print('Chunk length:', chunk_length)
        #print('No of frames:', no_of_frames)
        if chunk_length < 1:
            return skel_data_right_hand
        max_chunk_start = int(no_of_frames - chunk_length)
        chunk_start = np.random.randint(0, high=max_chunk_start) # high is exlusive
        chunk_end = chunk_start + chunk_length

        elbow_to_wrist_interp = self.get_sphereical_linear_interpolation(joint_elbow, joint_wrist, chunk_start, chunk_end)
        wrist_to_hand_interp = self.get_sphereical_linear_interpolation(elbow_to_wrist_interp, joint_hand, chunk_start, chunk_end)
        wrist_to_thumb_interp = self.get_sphereical_linear_interpolation(elbow_to_wrist_interp, joint_thumb, chunk_start, chunk_end)
        hand_to_tip_of_hand_interp = self.get_sphereical_linear_interpolation(wrist_to_hand_interp, joint_tip_of_hand, chunk_start, chunk_end)

        skel_data_right_hand[1] = elbow_to_wrist_interp
        skel_data_right_hand[2] = wrist_to_hand_interp
        skel_data_right_hand[3] = hand_to_tip_of_hand_interp
        skel_data_right_hand[4] = wrist_to_thumb_interp

        return skel_data_right_hand

    def get_sphereical_linear_interpolation(self, joint1, joint2, start_time, end_time):

        # Calculate the vectors from joint1 to joint2
        bone_start = joint2[:, start_time] - joint1[:, start_time] 
        bone_end = joint2[:, end_time] - joint1[:, end_time]

        if bone_end.all() == 0 or bone_start.all() == 0:
            return joint2

        num_frames = end_time - start_time

        start_unit_vector = bone_start / np.linalg.norm(bone_start)
        end_unit_vector = bone_end / np.linalg.norm(bone_end)

        rotation = R.align_vectors([end_unit_vector], [start_unit_vector])[0]
        key_rotations = R.concatenate([R.identity(), rotation])
        key_times = [0, 1]

        time = np.linspace(0, 1, num_frames)
        slerp_rotations = Slerp(key_times, key_rotations)
        interpolated_rotations = slerp_rotations(time)

        interp_bone_start = interpolated_rotations.apply(bone_start)
        interp_joint2 = joint1[:, start_time:end_time] + interp_bone_start.T

        interp_joint2_all_t = joint2.copy()
        interp_joint2_all_t[:, start_time:end_time] = interp_joint2

        return interp_joint2_all_t

    def reduce_frame_rate(self):

        N,C,T,V,M = self.data.shape
        data = np.zeros((N, C, T, V, M))
        FPS_drop = self.FPS / 30

        print('Degradation type:', self.structured_degradation_type)
        print('Frame rate:', self.FPS)
        print('FPS drop:', FPS_drop)

        for i in tqdm(range(len(self.label))):
            no_of_frames = int(self.length[i].item())
            chunk_length = int(no_of_frames-no_of_frames*FPS_drop)
            if chunk_length < 1:
                continue
            max_chunk_start = int(no_of_frames - chunk_length)
            chunk_start = np.random.randint(0, high=max_chunk_start) # high is exclusive

            if self.mitigation is True:
                data[i] = self.data[i]

                start = self.data[i, :,chunk_start, :, :]
                end = self.data[i, :,chunk_start+chunk_length, :, :]

                for j in range(0, chunk_length):
                    increment = (end-start) / chunk_length
                    data[i,:,chunk_start+j,:,:] = start + increment * j
            else:
                delete_indices = np.arange(chunk_start, chunk_start+chunk_length)
                arr = np.delete(self.data[i].copy(), delete_indices, axis=1)
                data[i, :, :arr.shape[1], :, :] = arr
                self.length[i] = no_of_frames - chunk_length
                
        self.data = data

    def apply_reduced_resolution(self, skel_data, no_of_frames):

        N, C, T, V, M = self.data.shape
        data = np.zeros((N, C, T, V, M))
        for i in tqdm(range(len(self.label))):
            if self.mitigation is True:
                no_of_frames = self.length[i]
                data[i,:,:int(no_of_frames):self.structured_res,:,:] = self.data[i, :, :int(no_of_frames):self.structured_res, :, :]
                
                x = np.arange(no_of_frames)
                y = x[:int(no_of_frames):self.structured_res]

                selected_frames = np.where(np.isin(x,y))[0]
                all_indices = np.arange(len(x))
                unselected_indices = np.setdiff1d(all_indices, selected_frames)

                sub_arrays = np.split(unselected_indices, np.flatnonzero(np.diff(unselected_indices.T)!=1) + 1)
                for sub_array in sub_arrays:
                    if len(x)-1 in sub_array:
                        continue
                    len_sub_array = len(sub_array)
                    if len_sub_array > 1:
                        for j in range(len_sub_array):
                            
                            if sub_array[len_sub_array-1]+1 == len(x):
                                end = self.data[i,:,sub_array[len_sub_array-1],:,:]
                            else:
                                end = self.data[i,:,sub_array[len_sub_array-1]+1,:,:]
                            start = self.data[i,:,sub_array[0]-1,:,:]
                            
                            inc = (end-start) / (len_sub_array+1)
                            data[i,:,sub_array[j],:,:] = inc * (j+1) + start
                    else:
                        if len(sub_array) == 0:
                            continue
                        data[i,:,sub_array[0]] = (self.data[i,:,sub_array[0]-1,:,:] + self.data[i,:,sub_array[0]+1,:,:]) / 2
            else:
                tmp = self.data[i,:,:int(self.length[i]):self.structured_res,:,:]
                data[i,:,:tmp.shape[1]:,:,:] = tmp
                self.length[i] = int(tmp.shape[1])

        self.data = data

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
        #self.data.setflags(write=1)  # enable write mode for mmap
        for i in range(len(self.label)):
            if self.length[i].item() <= 1:
                continue
            drop_indices = np.random.choice(range(1, int(self.length[i].item())), 
                                        size=int(self.drop_rate * int(self.length[i].item())), replace=False)
            drop_indices = np.sort(drop_indices)
            sub_arrays = np.split(drop_indices, np.flatnonzero(np.diff(drop_indices)!=1) + 1) # group them in consecutive groups
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
                    #print(len(sub_array), i)
                    if sub_array[0] == 299:
                        self.data[i,:,sub_array[0]] = self.data[i,:,sub_array[0]]
                    else:
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


