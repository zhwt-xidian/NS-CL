import torch
import numpy as np
import tensorly as tl
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
import tqdm
import scipy.signal as signal

tl.set_backend('pytorch')

class TuckerProduct(nn.Module):
    '''
    input_size = (channl * heigh * width)
    output_size = (channl * heigh * width)
    '''
    def __init__(self, input_size: tuple, output_size: tuple):
        super(TuckerProduct, self).__init__()
        self.input_size = list(input_size)
        self.output_size = list(output_size)

        #channel
        self.U2 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.output_size[0], self.input_size[0])), dtype=torch.float, requires_grad=True))
        #height
        self.U3 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.output_size[1], self.input_size[1])), dtype=torch.float, requires_grad=True))
        #width
        self.U4 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.output_size[2], self.input_size[2])), dtype=torch.float, requires_grad=True))

        #定义需要用到的层
        self.relu = nn.ReLU()

    def forward(self,input):
        out = tl.tenalg.mode_dot(input,self.U2, mode=1)
        out = tl.tenalg.mode_dot(out, self.U3, mode=2)
        out = tl.tenalg.mode_dot(out, self.U4, mode=3)

        out = self.relu(out)

        return out


class resnet_tucker(nn.Module):
    '''
    input_size = (channl * heigh * width)
    '''
    def __init__(self, input_size: tuple):
        super(resnet_tucker, self).__init__()
        self.input_size = list(input_size)

        #channel
        self.U2 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[0], self.input_size[0])), dtype=torch.float, requires_grad=True))
        #height
        self.U3 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[1], self.input_size[1])), dtype=torch.float, requires_grad=True))
        #width
        self.U4 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[2], self.input_size[2])), dtype=torch.float, requires_grad=True))

        #定义需要用到的层
        self.relu = nn.ReLU()

    def forward(self,input):
        out = tl.tenalg.mode_dot(input,self.U2, mode=1)
        out = tl.tenalg.mode_dot(out, self.U3, mode=2)
        out = tl.tenalg.mode_dot(out, self.U4, mode=3)
        out = self.relu(out)

        out = out + input
        return out


class resnet_tucker_2DCNN(nn.Module):
    def __init__(self,input_size,middel_size):
        super(resnet_tucker_2DCNN, self).__init__()
        self.input_size = list(input_size)
        self.cnn_ch = middel_size

        # channel
        self.U2 = torch.nn.Parameter(
            torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[0], self.input_size[0])), dtype=torch.float,requires_grad=True))
        # height
        self.U3 = torch.nn.Parameter(
            torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[1], self.input_size[1])), dtype=torch.float,requires_grad=True))
        # width
        self.U4 = torch.nn.Parameter(
            torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[2], self.input_size[2])), dtype=torch.float,requires_grad=True))
        #cnn layer
        self.cnn_2d1 = nn.Conv2d(self.input_size[0], self.cnn_ch, kernel_size=3, stride=1, padding=1)
        self.cnn_2d2 = nn.Conv2d(self.cnn_ch, self.input_size[0], kernel_size=1)

        #relu
        self.relu_tucker = nn.ReLU()
        self.relu_cnn1 = nn.ReLU()
        self.relu_cnn2 = nn.ReLU()

        self.LN_tuck = nn.LayerNorm([self.input_size[0],self.input_size[1], self.input_size[2]])
        self.LN_cnn1 = nn.LayerNorm([self.cnn_ch, self.input_size[1], self.input_size[2]])
        self.LN_cnn2 = nn.LayerNorm([self.input_size[0], self.input_size[1], self.input_size[2]])

    def forward(self, input):
        out = tl.tenalg.mode_dot(input, self.U2, mode=1)
        out = tl.tenalg.mode_dot(out, self.U3, mode=2)
        out = tl.tenalg.mode_dot(out, self.U4, mode=3)
        out = self.relu_tucker(out)
        out = self.LN_tuck(out)

        out2 = self.cnn_2d1(input)
        out2 = self.relu_cnn1(out2)
        out2 = self.LN_cnn1(out2)

        out2 = self.cnn_2d2(out2)
        out2 = self.relu_cnn2(out2)
        out2 = self.LN_cnn2(out2)

        out = out + out2 + input
        return out


class My_output_layer(torch.nn.Module):
    def __init__(self,h_in,w_in,c_in,category):
        super().__init__()
        self.hin = h_in
        self.win = w_in
        self.cin = c_in
        self.CLASS = category

        self.weight = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.CLASS,self.cin,self.hin,self.win)), dtype=torch.float, requires_grad=True))

    def forward(self,input):
        result = My_dot_FourByFout(input,self.weight)
        return result


class SupervisedComparativeLearning(nn.Module):
    '''
    代码用于有监督对比学习的编码器_收缩层
    input_size = (batch * classes * channl * heigh * width)
    output_size = (batch * classes * channl * heigh * width)
    '''
    def __init__(self, input_size: tuple, output_size: tuple):
        super(SupervisedComparativeLearning, self).__init__()
        self.input_size = list(input_size)
        self.output_size = list(output_size)

        #channel
        self.U2 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.output_size[0], self.input_size[0])), dtype=torch.float, requires_grad=True))
        #height
        self.U3 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.output_size[1], self.input_size[1])), dtype=torch.float, requires_grad=True))
        #width
        self.U4 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.output_size[2], self.input_size[2])), dtype=torch.float, requires_grad=True))

        #定义需要用到的层
        self.relu = nn.RReLU(0.1,0.2)

    def forward(self,input):
        out = tl.tenalg.mode_dot(input,self.U2,mode=2)
        out = tl.tenalg.mode_dot(out,self.U3,mode=3)
        out = tl.tenalg.mode_dot(out,self.U4,mode=4)

        out = self.relu(out)

        return out


class SupervisedComparativeLearning_res(nn.Module):
    '''
    代码用于有监督对比学习的编码器_残差层
    input_size = (batch * classes * channl * heigh * width)
    '''
    def __init__(self, input_size: tuple):
        super(SupervisedComparativeLearning_res, self).__init__()
        self.input_size = list(input_size)

        #channel
        self.U2 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[0], self.input_size[0])), dtype=torch.float, requires_grad=True))
        #height
        self.U3 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[1], self.input_size[1])), dtype=torch.float, requires_grad=True))
        #width
        self.U4 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.input_size[2], self.input_size[2])), dtype=torch.float, requires_grad=True))

        #定义需要用到的层
        self.relu = nn.RReLU(0.1,0.2)

    def forward(self,input):
        out = tl.tenalg.mode_dot(input,self.U2,mode=2)
        out = tl.tenalg.mode_dot(out,self.U3,mode=3)
        out = tl.tenalg.mode_dot(out,self.U4,mode=4)

        out = self.relu(out)

        out = out + input

        return out


class Cp_linear_2D(nn.Module):
    def __init__(self,input_size,output_size,rank):
        super(Cp_linear_2D, self).__init__()
        self.in_feature = input_size
        self.out_feature = output_size
        self.r = rank

        self.U1 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.in_feature, self.r)), dtype=torch.float, requires_grad=True))
        self.U2 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.out_feature, self.r)), dtype=torch.float, requires_grad=True))

        self.lam = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.r)), dtype=torch.float, requires_grad=True))
        self.bais = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.out_feature)), dtype=torch.float, requires_grad=True))

    def forward(self,x):
        temp = tl.kruskal_to_tensor((self.lam,[self.U1,self.U2]))
        temp = torch.mm(x,temp)
        temp = temp + self.bais

        return temp


class Cp_linear_4D(nn.Module):
    def __init__(self,h_in,w_in,c_in,category,rank):
        super(Cp_linear_4D, self).__init__()
        self.hin = h_in
        self.win = w_in
        self.cin = c_in
        self.CLASS = category
        self.r = rank

        self.U1 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.hin, self.r)), dtype=torch.float, requires_grad=True))
        self.U2 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.win, self.r)), dtype=torch.float, requires_grad=True))
        self.U3 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.cin, self.r)), dtype=torch.float, requires_grad=True))
        self.U4 = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.CLASS, self.r)), dtype=torch.float, requires_grad=True))


        self.lam = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, size=(self.r)), dtype=torch.float, requires_grad=True))


    def forward(self,x):
        temp = tl.kruskal_to_tensor((self.lam,[self.U4,self.U3,self.U2,self.U1]))
        temp = My_dot_FourByFout(x,temp)
        return temp

#________-------------------------------------------------------______________#
def zeros_normalization(data):
    '''
    本代码将输入数据(5-D)每一个谱段归一化为[0,1]区间
    :param data: size(point * class * channel * win * win)  数据每一个class取point点
    :return:size(point * class * channel * win * win)
    '''
    # 获取每个样本中每个通道的最大值
    max_num = torch.max(data, dim=-1)
    max_num = torch.max(max_num[0], dim=-1)
    max_num = torch.unsqueeze(max_num[0], dim=3)  # 整理数据维度，增加两个维度
    max_num = torch.unsqueeze(max_num, dim=4)

    # 获取每个样本中每个通道的最小值
    min_num = torch.min(data, dim=-1)
    min_num = torch.min(min_num[0], dim=-1)
    min_num = torch.unsqueeze(min_num[0], dim=3)  # 整理数据维度，增加两个维度
    min_num = torch.unsqueeze(min_num, dim=4)

    # 将每个样本每个通道归一化到[0,1]区间
    out = data - min_num
    out = out / (max_num - min_num)

    return out

def Z_score_normalization(data):
    '''
    本代码将输入数据(5-D)每一个谱段归一化为均值为 0，方差为 1 的数据
    :param data: size(point * class * channel * win * win)  数据每一个class取point点
    :return: size(point * class * channel * win * win)
    '''
    mean = torch.mean(data, dim=[3, 4])
    mean = torch.unsqueeze(mean, dim=3)  # 整理数据维度，增加两个维度
    mean = torch.unsqueeze(mean, dim=4)
    std = torch.std(data, dim=[3, 4])
    std = torch.unsqueeze(std, dim=3)  # 整理数据维度，增加两个维度
    std = torch.unsqueeze(std, dim=4)

    out = (data - mean) / std

    return out

def My_dot_FourByFout(x,y):
    batch = x.shape[0]
    category = y.shape[0]
    result = torch.mm(x.reshape(batch, -1), y.reshape(category, -1).T)
    return result


def nor(data_set):
    mean = np.mean(np.mean(data_set, axis=0), axis=0)
    std = np.std(np.std(data_set, axis=0), axis=0)
    data_set = (data_set - mean) / std
    return data_set


def normalize(in_tensor):
    tensor_mean = in_tensor.mean()
    tensor_std = in_tensor.std()

    out_tensor = (in_tensor - tensor_mean) / tensor_std
    return out_tensor


def image_Gaussian_conv(input_img):
    "预卷积运算，使用3*3高斯卷积"
    # kernal = np.array([[1,1,1],[1,1,1],[1,1,1]])  # 3*3 mean filter
    # kernal = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])    # 5*5 mean filter
    # kernal = np.array([[0.05,0.1,0.05],[0.1,0.4,0.1],[0.05,0.1,0.05]])  #3*3 Gaussian filter
    kernal = np.array([[1, 1, 2, 1, 1], [1, 3, 4, 3, 1], [2, 4, 8, 4, 2], [1, 3, 4, 3, 1],[1, 1, 2, 1, 1]]) # 5*5 Gaussian filter

    res = np.zeros_like(input_img)
    a, b, c = input_img.shape
    print("开始高斯滤波")
    bar = tqdm([i for i in range(c)])
    for step, index in enumerate(bar):
        temp = input_img[:, :, index]
        temp_conv = signal.convolve2d(temp, kernal, mode='same')
        temp_conv = signal.convolve2d(temp, kernal, mode='same')
        res[:, :, index] = temp_conv
        bar.desc = "正在对第{}/{}层进行滤波".format(index + 1, c)
        print("滤波结束")
        return res
