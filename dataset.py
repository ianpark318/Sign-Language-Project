from glob import glob
from tqdm import tqdm
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import yaml


class OurDataset(Dataset):
    def __init__(self, keyp_path):
        with open("configs/default.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.keyp_path = keyp_path

        self.inputs = []
        data_num = 50  # How many datas per words?
        class_num = cfg["num_classes"]
        t = []
        for i in range(class_num):
            t.append(torch.tensor([i] * data_num))

        self.labels = torch.cat(t)
        words = os.listdir(self.keyp_path)
        words.sort()
        for w in words:
            keypoint_list = glob(os.path.join(self.keyp_path, w) + "/*")
            keypoint_list.sort()
            for keypoint in tqdm(keypoint_list, desc=f"{w} keypoint loading"):
                with open(keypoint, "r") as f:
                    data = json.load(f)
                hands_keypoints = data["data"]
                hands_keypoints = torch.tensor(hands_keypoints)
                frames_len = hands_keypoints.shape[0]
                ids = np.round(np.linspace(0, frames_len - 1, 60))
                keypoint_sequence = []
                for i in range(60):
                    keypoint_sequence.append(
                        hands_keypoints[int(ids[i]), ...].unsqueeze(0)
                    )
                keypoint_sequence = torch.cat(keypoint_sequence, dim=0)
                keypoint_sequence = keypoint_sequence.unsqueeze(0)
                self.inputs.append(keypoint_sequence)

        self.inputs = torch.cat(self.inputs, dim=0)

    def __getitem__(self, idx):
        return {
            "input": self.inputs[idx, ...],
            "label": self.labels[idx],
            "frame_len": 60,
        }

    def __len__(self):
        return len(self.inputs)
