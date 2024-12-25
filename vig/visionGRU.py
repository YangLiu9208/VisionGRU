import torch
import torch.nn as nn
from functools import partial
from minGRU_pytorch import minGRU

expansion_factor= 2
dim= 288//4
path_drop = 0
path_drop_rate = 0.0
numblock = 14*4
blocksize = 16//4
layer_norm = 1
sample_length = 14**2

mlp_ratio = 4

dw = 1

num_epochs = 300

L = [2,2,8,2]
dataset = "i1k_"



class Tokenizer(nn.Module):
    def __init__(self, block_size, token_dim):
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
        n = x.size(0)

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.view(n, numblock*numblock, dim)

        ret['x'] = x
        return ret



class Block2DGRU(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1, nb=numblock, dep=-1):
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

        residual = x
        x = self.norm(x)

        n = x.size(0)

        if dw:
            x = x.view(n, self.numblock, self.numblock, self.dim).permute(0,3,1,2) # n c h w
            x = self.dw_conv(x)
            x = x.view(n, self.dim, self.numblock**2).permute(0,2,1)
        
        x_reshaped = x.view(-1, self.numblock, self.numblock, self.dim)
        x_2 = torch.flip(x_reshaped, dims=[1,2]).view(n, self.numblock ** 2, self.dim)
        x_2 = torch.jit.fork(self.min_gru_2, x_2)

        x_1 = self.min_gru_1(x)

        x_1 = self.dropout(x_1)
        x_2 = self.dropout(torch.jit.wait(x_2)).flip(dims=[1])

        x = x_1 + x_2
        x = self.projection(x)

        x = self.dropout(x)

        x = x + residual

        residual = x

        x = self.norm2(x)

        x = self.projection1(x)
        x = self.activation(x)
        x = self.projection2(x)

        x = x + residual

        ret['x'] = x

        return ret

class GRUBlocks(nn.Module):
    def __init__(self, input_dim, dropout_rate=path_drop_rate):
        super().__init__()
        self.layers = nn.ModuleList(nn.Sequential(
            *[Block2DGRU(input_dim, dropout_rate, nb) for _ in range(num_layer)]
        ) for num_layer, input_dim, nb in [(L[0], dim, numblock), (L[1], 2*dim, numblock//2), (L[2], 4*dim,numblock//4), (L[3], 8*dim,numblock//8)])
        self.norm1 = partial(nn.LayerNorm, eps=1e-6)(dim*2)
        self.norm2 = partial(nn.LayerNorm, eps=1e-6)(dim*4)
        self.norm3 = partial(nn.LayerNorm, eps=1e-6)(dim*8)
        self.conv1 = nn.Conv2d(dim,dim*2,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(dim*2,dim*4,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(dim*4,dim*8,kernel_size=3,stride=2,padding=1)


    def forward(self, ret):
        n, seq_len, _ = ret['x'].shape

        ret = self.layers[0](ret)
        x = ret['x']
        x = self.conv1(x.permute(0, 2, 1).view(n, dim, numblock, numblock)).permute(0, 2, 3, 1)
        x = self.norm1(x).view(n,numblock*numblock//4,2*dim)
        ret['x'] = x
        ret = self.layers[1](ret)
        x = ret['x']
        x = self.conv2(x.permute(0, 2, 1).view(n, 2*dim, numblock//2, numblock//2)).permute(0, 2, 3, 1)
        x = self.norm2(x).view(n, numblock * numblock//16, 4*dim)
        ret['x'] = x
        ret = self.layers[2](ret)
        x = ret['x']
        x = self.conv3(x.permute(0, 2, 1).view(n, 4*dim, numblock//4, numblock//4)).permute(0, 2, 3, 1)
        x = self.norm3(x).view(n, numblock * numblock//64, 8*dim)
        ret['x'] = x
        ret = self.layers[3](ret)
        x = ret['x']
        return x.mean([1])


class VisionGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.tokenizer = Tokenizer(blocksize, dim)
        self.snake_gru = GRUBlocks(dim)
        self.fc = nn.Linear(8*dim, num_classes)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.snake_gru(x)
        x = self.fc(x)
        return x

def initialize_model(num_classes, compile=True):
    model = VisionGRU(num_classes=num_classes)
    if compile:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    return model
