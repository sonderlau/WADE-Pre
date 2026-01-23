from torch import nn


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', group_norm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if group_norm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()
        

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', dropout_rate: float = 0.1): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.dropout(h)
        
        return h + self.res_conv(x)
    

class DilatedResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dilation:int = 1, groups = 8, kernel_size=3, padding_mode='zeros', dropout_rate: float = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, dim_out),
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, dim_out),
            nn.GELU()
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.block1(x)
        
        h = self.dropout(h)
        
        h = self.block2(h)
        
        
        return h + self.res_conv(x)