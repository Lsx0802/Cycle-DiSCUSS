import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

from torchvision import transforms
from torch.utils.data import DataLoader
from model import FeatureFusion2D3DModel, FeatureFusion3D3DModel
from mydataset import MyDataset_DiSCUSS

from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import mean_squared_error as MSE

from utils import apply_transform,output_to_homogeneous_matrix,crop
from tqdm import tqdm


# 训练模型
def train(model_us_SVR ,model_us_SVR2, model_us_ct3D, train_loader, optimizer1,optimizer2,optimizer3, num_epochs):
    for epoch in tqdm(range(num_epochs)):
        model_us_SVR.train()
        model_us_SVR2.train()
        model_us_ct3D.train()

        for batch_idx, (data_2d, data_3d_us,data_3d_ct, T_Base2US) in enumerate(train_loader):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            us_SVR_pred_pose = output_to_homogeneous_matrix(model_us_SVR(data_2d, data_3d_us))
            us_SVR_pred_image =crop(data_3d_us,us_SVR_pred_pose)
            loss_us_SVR = MSE(us_SVR_pred_pose, T_Base2US) + 1-SSIM(us_SVR_pred_image,data_2d)
            
            us_ct3D_pred_pose = output_to_homogeneous_matrix(model_us_ct3D(data_3d_us, data_3d_ct))
            T_CT2US = us_ct3D_pred_pose @ T_Base2US
            us_ct3D_pred_image =crop(data_3d_ct,T_CT2US)
            US_moved=apply_transform(data_3d_us,us_ct3D_pred_pose)
            loss_us_ct3D =1- SSIM(us_ct3D_pred_image,data_2d)+1-SSIM(us_SVR_pred_image,us_ct3D_pred_image)+1-SSIM(US_moved,data_3d_ct)
            
            us_SVR2_pred_pose = output_to_homogeneous_matrix(model_us_SVR2(data_2d, US_moved))
            us_SVR2_pred_image =crop(data_3d_us,us_SVR2_pred_pose)
            loss_us_SVR2 =MSE(us_SVR2_pred_pose, T_CT2US) + 1-SSIM(us_SVR2_pred_image,data_2d)\
                        +1-SSIM(us_SVR2_pred_image,us_SVR_pred_image) +1-SSIM(us_SVR2_pred_image, us_ct3D_pred_image)

            loss=loss_us_SVR + loss_us_ct3D + loss_us_SVR2

            
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                # 使用TensorBoard记录每个loss
                writer.add_scalar('Loss/train_us_SVR', loss_us_SVR.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/train_us_ct3D', loss_us_ct3D.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/train_us_SVR2', loss_us_SVR2.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/train_total', loss.item(), epoch * len(train_loader) + batch_idx)


def validate(model_us_SVR ,model_us_SVR2, model_us_ct3D, val_loader):
    model_us_SVR.eval()
    model_us_SVR2.eval()
    model_us_ct3D.eval()

    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data_2d, data_3d_us, data_3d_ct, T_Base2US) in enumerate(val_loader):

            us_SVR_pred_pose = output_to_homogeneous_matrix(model_us_SVR(data_2d, data_3d_us))
            us_SVR_pred_image =crop(data_3d_us,us_SVR_pred_pose)
            loss_us_SVR = MSE(us_SVR_pred_pose, T_Base2US) + 1-SSIM(us_SVR_pred_image,data_2d)
            
            us_ct3D_pred_pose = output_to_homogeneous_matrix(model_us_ct3D(data_3d_us, data_3d_ct))
            T_CT2US = us_ct3D_pred_pose @ T_Base2US
            us_ct3D_pred_image =crop(data_3d_ct,T_CT2US)
            US_moved=apply_transform(data_3d_us,us_ct3D_pred_pose)
            loss_us_ct3D = 1-SSIM(us_ct3D_pred_image,data_2d)+1-SSIM(us_SVR_pred_image,us_ct3D_pred_image)+1-SSIM(US_moved,data_3d_ct)
            
            us_SVR2_pred_pose = output_to_homogeneous_matrix(model_us_SVR2(data_2d, US_moved))
            us_SVR2_pred_image =crop(data_3d_us,us_SVR2_pred_pose)
            loss_us_SVR2 =MSE(us_SVR2_pred_pose, T_CT2US) + 1-SSIM(us_SVR2_pred_image,data_2d)\
                        +1-SSIM(us_SVR2_pred_image,us_SVR_pred_image) +1-SSIM(us_SVR2_pred_image, us_ct3D_pred_image)
            
            loss = loss_us_SVR + loss_us_ct3D + loss_us_SVR2
            total_loss += loss.item()

            # 使用TensorBoard记录每个loss
            writer.add_scalar('Loss/val_us_SVR', loss_us_SVR.item(), batch_idx)
            writer.add_scalar('Loss/val_us_ct3D', loss_us_ct3D.item(), batch_idx)
            writer.add_scalar('Loss/val_us_SVR2', loss_us_SVR2.item(), batch_idx)
            writer.add_scalar('Loss/val_total', loss.item(), batch_idx)



    average_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {average_loss:.4f}')

if __name__ == '__main__':
    # 初始化TensorBoard的SummaryWriter
    writer = SummaryWriter()

    # 定义超参数
    batch_size = 4
    learning_rate = 0.0001
    num_epochs = 10000

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载数据集
    data_set=MyDataset_DiSCUSS(root_dir='DiSCUSS/data',transform=transform)

    # train_dataset = data_set[0:len(data_set)*0.8]
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = data_set[len(data_set)*0.8:]
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = data_set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = data_set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型
    model_us_SVR = FeatureFusion2D3DModel(num_classes=12)
    model_us_SVR2 = FeatureFusion2D3DModel(num_classes=12)
    model_us_ct3D=  FeatureFusion3D3DModel(num_classes=12)

    # 定义损失函数和优化器

    optimizer1 = optim.Adam(model_us_SVR.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(model_us_SVR2.parameters(), lr=learning_rate)
    optimizer3 = optim.Adam(model_us_ct3D.parameters(), lr=learning_rate)


    train(model_us_SVR,model_us_SVR2, model_us_ct3D, train_loader, optimizer1,optimizer2,optimizer3, num_epochs)
    validate(model_us_SVR,model_us_SVR2, model_us_ct3D, val_loader)
