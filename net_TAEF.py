import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

class MLP_Block1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention1(nn.Module):

    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        n, b, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), qkv)
        dots = torch.einsum('hid,hjd->hij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('hij,hjd->hid', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out



class net_TAEF(nn.Module):
    def __init__(self, Bands=102, num_endmembers=10, heads=6, dropout=0.1, Batchsize=100):  # pavia
    # def __init__(self, Bands=64, num_endmembers=10, heads=8, dropout=0.1, Batchsize=100):  # MUUFL
        # heads needs to be divisible by Bands
        super(net_TAEF, self).__init__()

        self.nn1 = nn.Linear(Bands, num_endmembers,bias=False)
        self.dp1=nn.Dropout(dropout)

        self.nn2 = nn.Linear(num_endmembers, Bands,bias=False)
        self.dp2 = nn.Dropout(dropout)

        self.ln1=nn.LayerNorm(num_endmembers)
        self.sm1=nn.Softmax(dim=1)

        self.nn31 = nn.Linear(2 * Bands, Bands, bias=False)
        self.dp3 = nn.Dropout(dropout)

        self.msa1=Attention1(Bands, heads=heads, dropout=dropout)
        self.pos_embedding1 = nn.Parameter(torch.empty(Batchsize, Bands))
        torch.nn.init.normal_(self.pos_embedding1, std=.02)
        self.ln2 = nn.LayerNorm(Bands)
        self.ln3 = nn.LayerNorm(Bands)
        self.mlp1 = MLP_Block1(Bands, num_endmembers, dropout=dropout)

        torch.nn.init.xavier_normal_(self.nn1.weight)
        torch.nn.init.xavier_normal_(self.nn2.weight)
        torch.nn.init.xavier_normal_(self.nn31.weight)


    def forward(self, x, mask=None):

        x0 = x + self.pos_embedding1
        x0 = self.ln2(x0)
        x0 = self.msa1(x0, mask=mask)
        x10 = x + x0

        x1 = self.ln3(x10)
        x1 = self.mlp1(x1)


        z = self.nn1(x1)
        z = self.sm1(z)
        y = self.nn2(z)

        yx=torch.mul(x,y)
        p_2b=torch.cat([y,yx],1)
        p_b = self.nn31(p_2b)

        p = p_b

        Bands=p_b.size(dim=1)
        p_z1=torch.zeros(100,Bands)
        p_z1[:, Bands-1] = p_b[:, Bands-2]
        p_z1[:, 0:(Bands-1)] = p_b[:, 1:Bands]

        p_y1 = torch.zeros(100, Bands)
        p_y1[:, 0] = p_b[:, 1]
        p_y1[:, 1:Bands] = p_b[:, 0:(Bands - 1)]
        p = (p_b + p_z1 + p_y1) / 3

        p_z2 = torch.zeros(100, Bands)
        p_z2[:, Bands - 1] = p_b[:, Bands - 3]
        p_z2[:, Bands - 2] = p_b[:, Bands - 2]
        p_z2[:, 0:(Bands - 2)] = p_b[:, 2:Bands]

        p_y2 = torch.zeros(100, Bands)
        p_y2[:, 0] = p_b[:, 2]
        p_y2[:, 1] = p_b[:, 1]
        p_y2[:, 2:Bands] = p_b[:, 0:(Bands - 2)]
        p = (p_b + p_z1 + p_y1 + p_z2 + p_y2) / 5

        p_z3 = torch.zeros(100, Bands)
        p_z3[:, Bands - 1] = p_b[:, Bands - 4]
        p_z3[:, Bands - 2] = p_b[:, Bands - 3]
        p_z3[:, Bands - 3] = p_b[:, Bands - 2]
        p_z3[:, 0:(Bands - 3)] = p_b[:, 3:Bands]

        p_y3 = torch.zeros(100, Bands)
        p_y3[:, 0] = p_b[:, 3]
        p_y3[:, 1] = p_b[:, 2]
        p_y3[:, 2] = p_b[:, 1]
        p_y3[:, 3:Bands] = p_b[:, 0:(Bands - 3)]
        p = (p_b + p_z1 + p_y1 + p_z2 + p_y2 + p_z3 + p_y3) / 7

        x2=torch.div(torch.mul((1-p),y),1-torch.mul(p,y))

        return z,y,x2



if __name__ == '__main__':
    model = net_TAEF()
    model.eval()
    print(model)
    input = torch.randn(100,64)
    z, y,x2 = model(input)
    print(y.size())
    print(z.size())



