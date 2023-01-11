import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import torchgeometry as tgm
from glob import glob
import os.path as osp
import cv2
import random

marginal = 32

patch_size = 128

class MyMutiDataset(Dataset):
    def __init__(self, args, split='train'):
        
        if split == 'train':
            if args.dataset=='zurich':
                root_img1 = './datasets/Zurich/A'
                root_img2 = './datasets/Zurich/B'
            if args.dataset=='sar_vis':
                root_img1 = './datasets/sar_vis/vis'
                root_img2 = './datasets/sar_vis/sar'
        else:
            if args.dataset=='zurich':
                root_img1 = './datasets/Zurich/A'
                root_img2 = './datasets/Zurich/B'
            if args.dataset=='sar_vis':
                root_img1 = './datasets/sar_vis/vis'
                root_img2 = './datasets/sar_vis/sar'
            if args.dataset=='zurich_old':
                root_img1 = './datasets/Zurich_old/A'
                root_img2 = './datasets/Zurich_old/B'
        if args.dataset=='sar_vis':
            self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.tif')))
            self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.tif')))
        else:
            self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.png')))
            self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.png')))

        print(len(self.image_list_img1))

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        # print("img1:" + self.image_list_img1[index])
        # print("img2:" + self.image_list_img2[index])
        img1_name = self.image_list_img1[index]
        img2_name = self.image_list_img2[index]

        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])

        # resize图片到 192 x 192 大小
        img1 = cv2.resize(img1, (192, 192))
        img2 = cv2.resize(img2, (192, 192))

        # 四个点的坐标（已裁剪）
        top_left_point = (marginal, marginal)
        bottom_right_point = (marginal + patch_size, marginal + patch_size)

        top_left_point_cord = (marginal, marginal)
        bottom_left_point_cord = (marginal, marginal + patch_size - 1)
        bottom_right_point_cord = (patch_size + marginal - 1, marginal + patch_size - 1)
        top_right_point_cord = (marginal + patch_size - 1, marginal)
        four_points_cord = [top_left_point_cord, bottom_left_point_cord, bottom_right_point_cord, top_right_point_cord]

        # 考虑到数据集特征，考虑是否一个更适合数据集特征的扰动方式
        # try:
        perturbed_four_points_cord = []
        # 随机旋转+随机平移+随机小量非规则扰动
        rotate_dir = [[1,-1], [1,1], [-1,1], [-1,-1]]
        r_rotate = random.randint(-marginal/2, marginal/2)
        r_move = random.randint(-marginal/2, marginal/2)
        for i in range(4):
            # 随机扰动
            # t1 = random.randint(-marginal/8, marginal/8)
            # t2 = random.randint(-marginal/8, marginal/8)
            t1 = random.randint(-marginal, marginal)
            t2 = random.randint(-marginal, marginal)

            # 存储扰动后的坐标
            perturbed_four_points_cord.append((four_points_cord[i][0] + t1,
                                                 four_points_cord[i][1] + t2))

            # perturbed_four_points_cord.append((four_points_cord[i][0] + t1 + r_rotate * rotate_dir[i][0] + r_move,
            #                                     four_points_cord[i][1] + t2 + r_rotate * rotate_dir[i][1] + r_move))

        y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

        org = np.float32(four_points_cord)
        dst = np.float32(perturbed_four_points_cord)
        # 按扰动后的版本获得这个扰动对应的透视矩阵
        H = cv2.getPerspectiveTransform(org, dst)
        H_inverse = np.linalg.inv(H)
        # except:
        #     perturbed_four_points_cord = []
        #     for i in range(4):
        #         t1 = 32//(i+1)
        #         t2 = -32//(i+1)

        #         perturbed_four_points_cord.append((four_points_cord[i][0] + t1,
        #                                           four_points_cord[i][1] + t2))

        #     y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
        #     point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

        #     org = np.float32(four_points_cord)
        #     dst = np.float32(perturbed_four_points_cord)
        #     H = cv2.getPerspectiveTransform(org, dst)
        #     H_inverse = np.linalg.inv(H)
        
        # 对img2应用给定扰动的透视变换
        warped_image = cv2.warpPerspective(img2, H_inverse, (img1.shape[1], img1.shape[0]))

        # img1：在原img1上给定的128x128的范围裁剪
        img_patch_ori = img1[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0], :]
        # print("img1:" + str(img_patch_ori.shape))
        # img2：先应用扰动的变换再裁剪
        img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0],:]
        # print("img2:" + str(img_patch_pert.shape))

        # 计算flow
        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H).squeeze()
        diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((img1.shape[0], img1.shape[1]))
        diff_y_branch1 = diff_y_branch1.reshape((img1.shape[0], img1.shape[1]))

        pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch = np.zeros((patch_size, patch_size, 2))
        pf_patch[:, :, 0] = pf_patch_x_branch1
        pf_patch[:, :, 1] = pf_patch_y_branch1

        img_patch_ori = img_patch_ori[:, :, ::-1].copy()
        img_patch_pert = img_patch_pert[:, :, ::-1].copy()
        img1 = torch.from_numpy((img_patch_ori)).float().permute(2, 0, 1)
        img2 = torch.from_numpy((img_patch_pert)).float().permute(2, 0, 1)

        flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()

        ### homo
        four_point_org = torch.zeros((2, 2, 2))
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([128 - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, 128 - 1])
        four_point_org[:, 1, 1] = torch.Tensor([128 - 1, 128 - 1])

        four_point = torch.zeros((2, 2, 2))
        four_point[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0])
        four_point[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([128 - 1, 0])
        four_point[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, 128 - 1])
        four_point[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([128 - 1, 128 - 1])
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        four_point = four_point.flatten(1).permute(1, 0).unsqueeze(0)
        H = tgm.get_perspective_transform(four_point_org, four_point)
        H = H.squeeze()

        return img2, img1, flow, H

def fetch_dataloader(args, split='train'):

    train_dataset = MyMutiDataset(args, split)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=4, drop_last=False)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader