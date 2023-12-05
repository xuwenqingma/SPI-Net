import math

import numpy as np
import torch

# position = torch.randn([3,4,2], dtype=torch.float32)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
# print(position)
# print(position[:,-1:])
from torch import nn, concat

# Conv = nn.Conv1d(9, 12, kernel_size=2, stride=1, bias=False)
# x = torch.randn(32, 9, 96)#B x C x T ,把dim=1看做通道
# y = Conv(x)
# print(y.shape)#[32, 12, 8]
#
# print("_____________________________")
# linear = nn.Linear(96, 12)
# x1 = torch.randn(32, 9, 96)
# y1 = linear(x1)
# print(y1.shape)#[32, 9, 12]

# class Splitting(nn.Module):
#     def __init__(self):
#         super(Splitting, self).__init__()
#
#     def even(self, x):
#         return x[:, ::2, :]
#
#     def odd(self, x):
#         return x[:, 1::2, :]
#
#     def forward(self, x):
#         '''Returns the odd and even part'''
#         return (self.even(x), self.odd(x))
#
# spli = Splitting()
# x = torch.randn([3,4,2])
# print(x)
# even, odd = spli(x)
# print("even:",even)
# print("odd:",odd)

# nn.Sequential()#nn.Sequential要求模块间有衔接，实现了farward()方法
# Modul = nn.ModuleList()#nn.ModuleList则没有顺序性要求，并且也没有实现forward()方法
# print(Modul)

# class SCINet(nn.Module):
#     def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1, num_stacks = 1,
#                 num_levels = 3, num_decoder_layer = 1, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
#                  single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False):
#         super(SCINet, self).__init__()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
# x = np.Inf
# print(x)

# class LevelSCINet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x
#
#
# class SCINet_Tree(nn.Module):
#     def __init__(self, loss, current_level):
#         super().__init__()
#         self.loss = loss
#         self.current_level = current_level
#         self.working = LevelSCINet()
#         if current_level != 0:
#             self.SCINet_Tree_odd = SCINet_Tree(loss,current_level - 1)
#             self.SCINet_Tree_even = SCINet_Tree(loss,current_level - 1)
#
#
#     def forward(self, x):
#         self.working(x)
#         loss=self.current_level
#         if self.current_level == 0:
#             self.loss.append(loss)  # B x T x C
#         else:
#             self.loss.append(loss)
#             self.SCINet_Tree_even(x)
#             self.SCINet_Tree_odd(x)
#         return 0
#
#
# class EncoderTree(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss = []
#
#         self.st = SCINet_Tree(self.loss, current_level=2)
#
#     def forward(self, x):
#         self.st(x)
#         return self.loss,np.average(self.loss)
#
# et = EncoderTree()
# loss, avge_loss = et(2)
# print(loss,avge_loss)

# i = 4
# for t in range(i,10):
#     print(t)

# t = [1,2,3,4]
# print(len(t))

# def fun(x):
#     return (x+1,x+2)
#
# x = fun(1)
# print(type(x))
# print(x[0])

# x = range(2,10)
# for t in x:
#     print(t)

import torch

# 创建一个2x3的二维tensor
# x = torch.randn(2,4,3)
# print(x)

# 扩展维度并保持维度顺序不变
# y = torch.unsqueeze(x, dim=1)
# print(y.shape)
# print(y)
# 输出：torch.Size([2, 1, 1, 3])

x1 = torch.randn(2,1,3)
x2 = torch.randn(2,1,3)
x = torch.cat([x1,x2],dim=1)
print(x.shape)


#zhangfeng
#760640 python -u run_financial_modified.py --dataset_name electricity --window_size 96 --horizon 96 --hidden-size 8 --stacks 2--levels 3 --lr 9e-4 --dropout 0.25 --batch_size 32 --model_name ele_I96_o96_lr9e-4_bs32_dp0_h8_s2l3_w0.5_n4 --groups 321  --concat_len 0   --normalize 4 --long_term_forecast >ele_96_s2l3_lr9e-4_dila9_fft_dp0.25.log 2>&1 &
#749065 python -u run_financial_modified.py --dataset_name electricity --window_size 96 --horizon 192 --hidden-size 8 --stacks 2 --levels 3 --lr 9e-4 --dropout 0 --batch_size 32 --model_name ele_I96_o192_lr9e-4_bs32_dp0_h8_s2l3_w0.5_n4 --groups 321  --concat_len 0   --normalize 4 --long_term_forecast >ele_192_s2l3_lr9e-4_dila3_fft_dp0_m9.log 2>&1 &
#750283 python -u run_financial_modified.py --dataset_name electricity --window_size 96 --horizon 336 --hidden-size 8 --stacks 2 --levels 3 --lr 9e-4 --dropout 0 --batch_size 32 --model_name ele_I168_o336_lr9e-4_bs32_dp0_h8_s2l3_w0.5_n4 --groups 321  --concat_len 0   --normalize 4 --long_term_forecast >ele_336_s2l3_lr9e-4_dila9_fft_dp0.log 2>&1 &

#750436 python run_ETTh_modified.py --data ETTm1 --features M  --seq_len 672 --label_len 288 --pred_len 288 --hidden-size 4 --stacks 1 --levels 5 --lr 1e-5 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I672_O288_lr1e-5_bs32_dp0.5_h0.5_s1l5
#753179 python run_ETTh_modified.py --data ETTm1 --features M  --seq_len 384 --label_len 96 --pred_len 96 --hidden-size 0.5 --stacks 2 --levels 4 --lr 5e-5 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I384_O96_lr5e-5_bs32_dp0.5_h0.5_s2l4

#760946 nohup python -u run_ETTh_modified.py --data ETTm1 --features M --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --levels 3 --lr 0.001 --batch_size 32 --dropout 0.25 --model_name ettm1_M_I48_O24_lr1e-3_bs16_dp0.25_h8_s1l3 >m1_24_dila9_s1l3_lr0.001_dp0.25.log 2>&1 &
#761003 nohup python -u run_ETTh_modified.py --data ETTm1 --features M --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --levels 3 --lr 0.005 --batch_size 32 --dropout 0.25 --model_name ettm1_M_I48_O24_lr1e-3_bs16_dp0.25_h8_s1l3 >m1_24_dila9_s1l3_lr0.005_dp0.25.log 2>&1 &


#nohup python -u run_ETTh_modified.py --data ETTh2 --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 4 --stacks 1 --levels 5 --lr 5e-4 --batch_size 128 --dropout 0.5 --model_name etth2_M_I736_O720_lr5e-4_bs128_dp0.5_h4_s1l5 >h2_720_dila9_s1l5_lr5e-4_dp0.5.log 2>&1 &
#
#1217913 nohup python -u run_ETTh_modified.py --data ETTm1 --features M  --seq_len 672 --label_len 288 --pred_len 288 --hidden-size 4 --stacks 1 --levels 5 --lr 1e-5 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I672_O288_lr1e-5_bs32_dp0.5_h0.5_s1l5 >m1_288_dila9_s1l5_lr1e-5_dp0.5.log 2>&1 &




#xwq
#2703929 nohup python -u run_ETTh_modified.py --data ETTm1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 2 --levels 4 --lr 0.001 --batch_size 16 --dropout 0.5 --model_name ettm1_M_I96_O48_lr1e-3_bs16_dp0.5_h4_s2l4 >m1_48_dila9_s2l4_lr0.001_dp0.25.log 2>&1 &
#2704007 nohup python -u run_ETTh_modified.py --data ETTm1 --features M --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --levels 3 --lr 0.005 --batch_size 32 --dropout 0.25 --model_name ettm1_M_I48_O24_lr1e-3_bs16_dp0.25_h8_s1l3 >m1_24_dila9_s1l3_lr0.005_dp0.25.log 2>&1 &
#