import os
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from utils import load_image,load_volume,load_pose


class MyDataset2D3D(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir_us_video = os.path.join(root_dir,'us_video')
        self.root_dir_us_volume = os.path.join(root_dir,'us_volume')
        self.root_dir_us_pose = os.path.join(root_dir,'pose')

        self.transform = transform
        self.video_files = sorted([f for f in os.listdir(self.root_dir_us_video) if f.endswith('.jpg') ])   
        #假设image的命名为0_0.jpg,0_1.jpg...
        #第一个0是volume的索引，第二个0是image的索引
        self.volume_files = sorted([f for f in os.listdir(self.root_dir_us_volume) if f.endswith('.nii.gz')])
        self.pose_files = sorted([f for f in os.listdir(self.root_dir_us_pose) if f.endswith('.csv')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir_us_video, self.video_files[idx])
        volume_index= self.video_files[idx].split('.')[0].split('_')
        image_index= self.video_files[idx].split('.')[1].split('_')
        volume_path = os.path.join(self.root_dir_us_volume, self.volume_files[volume_index])
        pose_path = os.path.join(self.root_dir_us_pose, self.pose_files[idx])

        image = load_image(image_path)
        volume = load_volume(volume_path)
        pose_list = load_pose(pose_path)
        pose=pose_list[image_index]

        if self.transform:
            image = self.transform(image)
            volume = self.transform(volume)
            pose = self.transform(pose)

        return image, volume, pose

class MyDataset3D3D(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir_volume_ct = os.path.join(root_dir,'ct_volume')
        self.root_dir_volume_us = os.path.join(root_dir,'us_volume')

        self.transform = transform
        self.volume_files_ct = sorted([f for f in os.listdir(self.root_dir_volume_ct) if f.endswith('.nii.gz')])
        self.volume_files_us = sorted([f for f in os.listdir(self.root_dir_volume_us) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        volume_path_ct = os.path.join(self.root_dir_volume_ct, self.volume_files_ct[idx])
        volume_path_us = os.path.join(self.root_dir_volume_us, self.volume_files_us[idx])

        volume_ct = load_volume(volume_path_ct)
        volume_us = load_volume(volume_path_us)

        if self.transform:
            volume_ct = self.transform(volume_ct)
            volume_us = self.transform(volume_us)

        return volume_ct, volume_us


class MyDataset_DiSCUSS(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir_us_video = os.path.join(root_dir,'us_video')
        self.root_dir_us_volume = os.path.join(root_dir,'us_volume')
        self.root_dir_us_pose = os.path.join(root_dir,'pose')
        self.root_dir_ct_volume = os.path.join(root_dir,'ct_volume')

        self.transform = transform
        self.video_files = sorted([f for f in os.listdir(self.root_dir_us_video) if f.endswith('.jpg') ])   
        #假设image的命名为0_0.jpg,0_1.jpg...
        #第一个0是volume的索引，第二个0是image的索引
        self.volume_files_ct = sorted([f for f in os.listdir(self.root_dir_ct_volume) if f.endswith('.nii.gz')])
        self.volume_files_us = sorted([f for f in os.listdir(self.root_dir_us_volume) if f.endswith('.nii.gz')])
        self.pose_files = sorted([f for f in os.listdir(self.root_dir_us_pose) if f.endswith('.csv')])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir_us_video, self.video_files[idx])
        volume_index= int(self.video_files[idx].split('.')[0].split('_')[0])
        image_index= int(self.video_files[idx].split('.')[0].split('_')[1])
        volume_path_us = os.path.join(self.root_dir_us_volume, self.volume_files_us[volume_index])
        volume_path_ct = os.path.join(self.root_dir_ct_volume, self.volume_files_ct[volume_index])
        pose_path = os.path.join(self.root_dir_us_pose, self.pose_files[volume_index])

        image = load_image(image_path)
        volume_us = load_volume(volume_path_us)
        volume_ct = load_volume(volume_path_ct)
        pose_list = load_pose(pose_path)
        pose=pose_list[image_index]

        if self.transform:
            image = self.transform(image)
            volume_us = self.transform(volume_us).unsqueeze(0)
            volume_ct = self.transform(volume_ct).unsqueeze(0)
            pose = self.transform(pose)

        return image, volume_us, volume_ct,pose