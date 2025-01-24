
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

import SimpleITK as sitk

def apply_transform(volume, transform_matrix):
    # 创建一个SimpleITK变换对象
    transform = sitk.AffineTransform(transform_matrix)

    # 使用变换对象对体积数据进行重采样
    resampled_volume = sitk.Resample(volume, volume.GetSize(), transform, sitk.sitkLinear, volume.GetOrigin(), volume.GetSpacing(), volume.GetDirection(), 0.0, volume.GetPixelID())

    return resampled_volume

def output_to_homogeneous_matrix(output):
    """
    将模型的输出转换为齐次矩阵。

    参数:
    output (torch.Tensor): 模型的输出，形状为 (12,)

    返回:
    torch.Tensor: 齐次矩阵，形状为 (4, 4)
    """
    # 确保输出是一个张量
    output = torch.tensor(output)

    # 将输出重塑为3x4的矩阵
    matrix_3x4 = output.view(3, 4)

    # 创建一个4x4的齐次矩阵，最后一行设置为 [0, 0, 0, 1]
    homogeneous_matrix = torch.zeros(4, 4)
    homogeneous_matrix[:3, :] = matrix_3x4
    homogeneous_matrix[3, 3] = 1

    return homogeneous_matrix

def crop(data, rotation_matrix):
    # 获取图像的间距、范围和原点
    spacing = data.GetSpacing()
    extents = data.GetSize()
    origin = data.GetOrigin()

    # 创建一个仿射变换对象
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rotation_matrix.flatten())
    transform.SetTranslation(origin)

    # 设置重采样的参数
    resampled_image = sitk.Resample(data, extents, transform, sitk.sitkLinear, origin, spacing, data.GetDirection(), 0.0, data.GetPixelID())

    return resampled_image

def get_homogeneous_transform(pose):
    """
    根据位置和旋转向量获取齐次变换矩阵。
    """
    rotation_matrix=R.from_rotvec(pose[3:]).as_matrix()
    # 创建4x4齐次变换矩阵
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] =pose[0:3]*1000
    
    return homogeneous_matrix
def load_image(image_path):
    # 使用SimpleITK读取图像
    image = sitk.ReadImage(image_path)
    
    # 将SimpleITK图像转换为NumPy数组
    array_data = sitk.GetArrayFromImage(image)
    
    # 获取图像的维度
    dimensions = image.GetSize()
    
    # 调整数组的形状以匹配图像的维度
    array_data = array_data.reshape(dimensions[1], dimensions[0], -1)
    
    # 如果图像是灰度图像，移除多余的维度
    if array_data.shape[2] == 1:
        array_data = array_data.squeeze(axis=2)
    
    return array_data


import SimpleITK as sitk
import numpy as np

def load_volume(volume_path):
    """
    读取NIfTI格式的体积数据，并将其转换为NumPy数组。

    参数:
    volume_path (str): NIfTI文件的路径。

    返回:
    np.ndarray: 转换后的NumPy数组。
    """

    # 使用SimpleITK读取NIfTI文件
    reader = sitk.ReadImage(volume_path)
    
    # 将SimpleITK图像转换为NumPy数组
    volume_array = sitk.GetArrayFromImage(reader)
    
    # 调整数组的形状以匹配体积的维度
    volume_array = volume_array.transpose((2, 1, 0))  # 调整数组形状以匹配体积维度
    
    return volume_array



def load_pose(pose_path):

    T_TCP2US=[[0, 0 ,-1 , 0],
            [-1 , 0 , 0 , 0],
            [0 , 1, 0 , 0],
            [ 0 ,0 , 0  ,1]]
    

    pose_list=[]
    df = pd.read_csv(pose_path,header=None)
    for i in range(0, len(df)):
        a=df.iloc[i].values
        T_base2TCP=get_homogeneous_transform(a)
        T_TCP2US=T_base2TCP @ T_TCP2US
        pose_list.append(T_TCP2US)
    return pose_list