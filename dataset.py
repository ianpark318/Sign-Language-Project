from glob import glob
from tqdm import tqdm
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class KSLDataset(Dataset):
    def __init__(self, is_train, morp_path, keyp_path):
        self.is_train = is_train
        self.morp_path = morp_path
        self.keyp_path = keyp_path
        
        self.inputs = []
        self.labels = []

        train_user = ['01', '02', '03', '04', '05', '06', '07', '08', \
                      '09', '10', '11', '12', '13', '14', '15', '16']
        val_user = ['17', '18']
        
        if is_train:
            user_list = train_user
            status = 'Train'
        else:
            user_list = val_user
            status = 'Validation'
        
        user_list.sort()

        # for user in user_list:
        #     user_morp_path = os.path.join(self.morp_path, user)
        #     morpheme_list = glob(user_morp_path + '/*')
        #     morpheme_list.sort()
        #     for morpheme in tqdm(morpheme_list, desc=f'{status} user {user} morpheme loading'):
        #         with open(morpheme) as json_file:
        #             data = json.load(json_file)
        #         l = data['data'][0]['attributes'][0]['name']
        #         self.labels.append(l)
        
        user1_label = []
        with open('labels.json','r') as f:
            dict_json = json.load(f)
            user1_label.extend(dict_json['labels'])
        
        if is_train:
            self.labels = user1_label * 16
        else:
            self.labels = user1_label * 2

        for user in user_list:
            user_keyp_path = os.path.join(self.keyp_path, user)
            keypoint_list = glob(user_keyp_path + '/*')
            keypoint_list.sort()
            for keypoint in tqdm(keypoint_list, desc=f'{status} user {user} keypoint loading'):
                frames = glob(keypoint + '/*')
                frames.sort()
                length = len(frames)
                ids = np.round(np.linspace(0, length - 1, 60))
                keypoint_sequence = []
                for i in range(60):
                    with open(frames[int(ids[i])]) as json_file:
                        data = json.load(json_file)
                    kp = []
                    kp.extend(data['people']['hand_left_keypoints_2d'])
                    kp.extend(data['people']['hand_right_keypoints_2d'])
                    keypoint_sequence.append(kp)
                self.inputs.append(keypoint_sequence)

        print(len(self.inputs), len(self.labels))

        self.inputs = torch.tensor(self.inputs)
        self.inputs = self.inputs.reshape(-1, 60, 42, 3)[..., :-1]
        
    def __getitem__(self, idx): 
        return {'input': self.inputs[idx, ...], 'label': self.labels[idx], 'frame_len': 60}
        
    def __len__(self): 
        return len(self.inputs)