import numpy as np
import os

import argparse
from utils import *

import scipy.io as io

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataset_custom as datasets
# import datasets_4cor_img as datasets
from network import IHN
from evaluate import validate_process

import numpy as np
import time

SUM_FREQ = 20
VAL_FREQ = 250

# 用于调整数值精度，加速训练
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# learning_rate = 2.5e-4

MAX_FLOW = 400

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
    def _print_training_status(self):
        # metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        
        # print the training status
        print(training_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        self.writer.add_scalar("loss", self.running_loss, self.total_steps)
        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.total_steps)

        # for k in self.running_loss:
        #     self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
        #     self.running_loss[k] = 0.0
    def push(self, loss):
        self.total_steps += 1

        self.running_loss = loss

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
    

def sequence_loss(flow_preds, flow_gts, gamma=0.85, iters_lev0=6):
    """ Loss function defined over sequence of flow predictions """

    flow_loss = 0.0
    n_predictions = len(flow_preds)

    for k in range(n_predictions-1):
        i_weight = gamma**(n_predictions - k - 1)
        i_loss = (flow_preds[k+1] - flow_gts).abs()
        flow_loss += i_weight * i_loss.mean()

    return flow_loss

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # 更新器设置
    # 正常流程：OneCycle
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # 继续训练：
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.0001, cycle_momentum=False, anneal_strategy='linear')
    

    return optimizer, scheduler

def train(args):

    print("start training")
    # model = nn.DataParallel(IHN(args), device_ids=args.gpus)
    model = IHN(args)

    # 继续训练的设置
    if args.restore_ckpt is not None:
       model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
       print("model restored:" + str(args.restore_ckpt))

    model.cuda()
    model.train()
    
    train_loader = datasets.fetch_dataloader(args, 'train')
    optimizer, scheduler = fetch_optimizer(args, model)

    # 引入GradScaler的误差调整
    scaler = GradScaler()
    logger = Logger(model, scheduler)


    total_steps = 0

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            print("training step:" + str(total_steps))
            optimizer.zero_grad()
            img1, img2, flow_gt, H = [x.cuda() for x in data_blob]

            four_pred, predictions = model(img1, img2, iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1)

            # flow_gt = flow_gt.squeeze(0)
            # print("flow_prediction:" + str(flow_prediction))
            flow_4cor = torch.zeros(four_pred.shape[0], 2, 2, 2).cuda()
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            # print("flow_4cor:" + str(flow_4cor))
            
            loss = sequence_loss(predictions, flow_4cor, args.gamma, args.iters_lev0)

            print("loss:" + str(loss))

            # 反向传播与学习率更新
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(loss)

            # 隔 VAL_FREQ 次进行验证过程，暂存ckpt并保存当前正确率
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                results.update(validate_process(model, args))
                logger.write_dict(results)
                
                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
    
    logger.close()
    # 保存参数
    PATH = 'checkpoints/{}.pth'.format(args.name)
    torch.save(model.state_dict(), PATH)

    return PATH
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='IHN', help="name your experiment")

    parser.add_argument('--dataset', type=str, default='zurich', help='dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--clip', type=float, default=1.0)


    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=3)
    parser.add_argument('--lev0', default=False, action='store_true',
                        help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true',
                        help='warp once')
    parser.add_argument('--weight', default=False, action='store_true',
                        help='weight')
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')

    # Adam优化器参数
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    train(args)



        

        
    

