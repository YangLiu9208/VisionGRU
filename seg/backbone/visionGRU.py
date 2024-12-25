import math
import threading

from mmseg.registry import MODELS
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from minGRU_pytorch import minGRU
from torch.nn import Linear, Identity
from torchvision import transforms, models
torch.set_float32_matmul_precision('high')

expansion_factor= 2 
dim= 288 
path_drop = 0 
path_drop_rate = 0.0 
blocksize = 16
layer_norm = 1 

mlp_ratio = 4

dw = 1
two = 1

num_epochs = 300

down_sample = 1
L = [2,2,8,2]
dataset = "i1k_"


dim//=4


class Tokenizer(nn.Module):
    def __init__(self, block_size, token_dim,):
        super().__init__()
        self.block_size = block_size
        self.token_dim = token_dim
        self.conv1 = nn.Conv2d(3,token_dim//2,3,2,1)
        self.conv2 = nn.Conv2d(token_dim//2,token_dim,3,2,1)
        self.gelu = nn.GELU()
        self.norm1 = partial(nn.LayerNorm, eps=1e-6)(token_dim//2)
        self.norm2 = partial(nn.LayerNorm, eps=1e-6)(token_dim)
        self.dropout = nn.Dropout(0.3)

    
    def forward(self, ret):
        if not isinstance(ret, dict):
            ret = {'x':ret}
        x = ret['x']
        n, c, h, w = x.shape

        x = self.conv1(x)
        h = math.ceil(h/2)
        w = math.ceil(w/2)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.conv2(x)
        h = math.ceil(h/2)
        w = math.ceil(w/2)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.view(n, h*w, dim)

        ret['h'] = h
        ret['w'] = w
        ret['x'] = x

        ret['tok0'] = x
        return ret  



class Block2DGRU(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1, nb=None, dep=-1):
        super().__init__()
    
        self.min_gru_1 = minGRU(dim=input_dim, expansion_factor=expansion_factor)
        self.min_gru_2 = minGRU(dim=input_dim, expansion_factor=expansion_factor)
        self.projection = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if path_drop else nn.Identity()
        self.projection1 = nn.Linear(input_dim, input_dim*mlp_ratio)
        self.activation = nn.GELU()
        self.projection2 = nn.Linear(input_dim*mlp_ratio, input_dim)
        self.norm2 = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()


        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.dim = input_dim
        self.numblock = nb
        self.dep = dep

        if dw:
            self.dw_conv = nn.Conv2d(in_channels=self.dim,out_channels=self.dim,kernel_size=3,stride=1,padding=1,groups=self.dim)
    def forward(self, ret):
        x = ret['x']
        h, w = ret['h'], ret['w']

        residual = x
        x = self.norm(x)

        n = x.size(0)


        if dw:
            x = x.view(n, h, w, self.dim).permute(0,3,1,2) # n c h w
            x = self.dw_conv(x)
            x = x.view(n, self.dim, h*w).permute(0,2,1)
        
        x_reshaped = x.view(-1, h, w, self.dim)
        x_2 = torch.flip(x_reshaped, dims=[1,2]).view(n,  h*w, self.dim)

        x_2 = torch.jit.fork(self.min_gru_2, x_2)

        x_1 = self.min_gru_1(x)

        x_1 = self.dropout(x_1)
        x_2 = self.dropout(torch.jit.wait(x_2)).flip(dims=[1])

        x = x_1 + x_2
        x = self.projection(x)

        x = x + residual

        residual = x

        x = self.norm2(x)

        x = self.projection1(x)
        x = self.activation(x)
        x = self.projection2(x)

        x = x + residual

        ret['x'] = x
        ret[f'tok{self.dep}']=x
        return ret

class Blocks2DGRU(nn.Module):
    def __init__(self, input_dim, dropout_rate=path_drop_rate):
        super().__init__()
        self.layers = nn.ModuleList(nn.Sequential(
            *[Block2DGRU(input_dim, dropout_rate) for _ in range(num_layer)]
        ) for num_layer, input_dim in [(L[0], dim), (L[1], 2*dim), (L[2], 4*dim,), (L[3], 8*dim,)])
        self.norm1 = partial(nn.LayerNorm, eps=1e-6)(dim*2)
        self.norm2 = partial(nn.LayerNorm, eps=1e-6)(dim*4)
        self.norm3 = partial(nn.LayerNorm, eps=1e-6)(dim*8)
        self.conv1 = nn.Conv2d(dim,dim*2,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(dim*2,dim*4,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(dim*4,dim*8,kernel_size=3,stride=2,padding=1)

    def forward(self, ret):
        n, seq_len, _ = ret['x'].shape

        B = n
        ret = self.layers[0](ret)
        x = ret['x']
        h, w = ret['h'], ret['w']
        x1=x.view(B, h, w, dim).permute(0,3,1,2)
        x = self.conv1(x.permute(0, 2, 1).view(B, dim, h, w)).permute(0, 2, 3, 1)
        h = math.ceil(h/2)
        w = math.ceil(w/2)
        x = self.norm1(x).view(B,h*w,2*dim)

        ret['h'] = h
        ret['w'] = w
        ret['x'] = x
        ret = self.layers[1](ret)
        x = ret['x']
        h, w = ret['h'], ret['w']
        x2=x.view(B, h, w, dim*2).permute(0,3,1,2)
        x = self.conv2(x.permute(0, 2, 1).view(B, 2*dim, h, w)).permute(0, 2, 3, 1)
        h = math.ceil(h/2)
        w = math.ceil(w/2)
        x = self.norm2(x).view(B, h * w, 4*dim)

        ret['h'] = h
        ret['w'] = w
        ret['x'] = x
        ret = self.layers[2](ret)
        x = ret['x']
        h, w = ret['h'], ret['w']
        x3=x.view(B, h, w, dim*4).permute(0,3,1,2)
        x = self.conv3(x.permute(0, 2, 1).view(B, 4*dim, h, w)).permute(0, 2, 3, 1)
        h = math.ceil(h/2)
        w = math.ceil(w/2)
        x = self.norm3(x).view(B, h * w, 8*dim)

        ret['h'] = h
        ret['w'] = w
        ret['x'] = x
        ret = self.layers[3](ret)
        x = ret['x']
        h, w = ret['h'], ret['w']
        x4=x.view(B, h, w, dim*8).permute(0,3,1,2)
        return x1,x2,x3,x4

class VisionGRUCore(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.tokenizer = Tokenizer(blocksize, dim)
        self.snake_gru = Blocks2DGRU(dim)

    def forward(self, x,):
        x = self.tokenizer(x)  
        x = self.snake_gru(x)  
        return x
    
@MODELS.register_module()
class VisionGRU(nn.Module):
    def __init__(self, pretrained = None, compile=True, ti=True,*args,**kargs):
        super().__init__()
        if not ti:
            global dim
            dim=416//4
            L[2]=15

        self.vig = VisionGRUCore()
        if pretrained:
            import torch
            self.vig.load_state_dict(torch.load(pretrained))
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if compile:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.vig = torch.compile(self.vig)

    def forward(self, x):
        return self.vig(x)