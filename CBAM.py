# Convolutional Block Attention Module
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

class CAM(nn.Module):
  def __init__(self, dim, radio):
    super(CAM, self).__init__()
    self.mlp = nn.Sequential(
      Rearrange('b c h w -> b h w c'),
      nn.Linear(dim, dim//radio),
      nn.ReLU(),
      nn.Linear(dim//radio, dim),
      Rearrange('b h w c -> b c h w')
    )
  def forward(self, f):
    return F.sigmoid(self.mlp(torch.mean(f, dim=(2, 3), keepdim=True)) + self.mlp(torch.max_pool2d(f,kernel_size=(f.shape[2], f.shape[3]))))
  
class SAM(nn.Module):
  def __init__(self, dim):
    super(SAM, self).__init__()
    self.maxPool = nn.MaxPool3d((dim, 1, 1))
    self.avgPool = nn.AvgPool3d((dim, 1, 1))
    self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
  def forward(self, f):
    return F.sigmoid(self.conv(torch.cat([self.maxPool(f), self.avgPool(f)], dim=1)))

class CBAM(nn.Module):
  def __init__(self, dim, radio):
    super(CBAM, self).__init__()
    self.cam = CAM(dim, radio)
    self.sam = SAM(dim)
  def forward(self, f):
    f = f * self.cam(f)
    f = f * self.sam(f)
    return f

if __name__ == "__main__":
  img = torch.randn(1, 3, 224, 224)
  net = CBAM(3, 2)
  y = net(img)
  print(y.shape, y)