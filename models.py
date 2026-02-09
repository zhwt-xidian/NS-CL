import torch
import torch.nn as nn
import torch.nn.functional as F
import tnnLIB 

class NSCL_Encoder(nn.Module):
    def __init__(self, in_ch, n_classes, patch_size=7):
        super().__init__()
        # 回归纯净版 Backbone，保留特征的原始判别力
        self.feature = nn.Sequential(
            tnnLIB.SupervisedComparativeLearning(input_size=(in_ch, 7, 7), output_size=(64, 11, 11)),
            nn.RReLU(),
            tnnLIB.SupervisedComparativeLearning_res(input_size=(64, 11, 11)),
            nn.RReLU(),
            tnnLIB.SupervisedComparativeLearning(input_size=(64, 11, 11), output_size=(128, 9, 9)),
            nn.RReLU(),
            tnnLIB.SupervisedComparativeLearning_res(input_size=(128, 9, 9)),
            nn.RReLU(),
            tnnLIB.SupervisedComparativeLearning(input_size=(128, 9, 9), output_size=(64, 7, 7))
        )
        self.flatten_dim = 64 * 7 * 7
        self.dropout = nn.Dropout(0.5)
        self.projector = nn.Sequential(nn.Linear(self.flatten_dim, 512), nn.ReLU(), nn.Linear(512, 128))
        self.classifier = nn.Linear(self.flatten_dim, n_classes) 

    def forward(self, x, mode='feat'):
        if x.dim() == 4: x = x.unsqueeze(1)
        elif x.dim() == 3: x = x.unsqueeze(0).unsqueeze(0)
            
        # 纯净前向，速度极快
        x = self.feature(x)
        f = torch.flatten(x, 1) 
        
        if mode == 'feat': return f
        elif mode == 'proj': return self.projector(f)
        elif mode == 'class': return self.classifier(self.dropout(f))