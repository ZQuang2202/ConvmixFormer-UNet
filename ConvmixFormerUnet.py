import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torchvision
import einops
from einops import rearrange, reduce

class PASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=4, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        if output_stride == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 4, 6, 10]
        elif output_stride == 2:
            dilations = [1, 12, 24, 36]
        elif output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 1:
            dilations = [1, 16, 32, 48]
        else:
            raise NotImplementedError
        self._norm_layer = BatchNorm
        self.silu = nn.SiLU(inplace=True)
        self.conv1 = self._make_layer(inplanes, inplanes // 4)
        self.conv2 = self._make_layer(inplanes, inplanes // 4)
        self.conv3 = self._make_layer(inplanes, inplanes // 4)
        self.conv4 = self._make_layer(inplanes, inplanes // 4)
        self.atrous_conv1 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[0], padding=dilations[0],groups=inplanes//4)
        self.atrous_conv2 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[1], padding=dilations[1],groups=inplanes//4)
        self.atrous_conv3 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[2], padding=dilations[2],groups=inplanes//4)
        self.atrous_conv4 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[3], padding=dilations[3],groups=inplanes//4)
        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, inplanes)

    def _make_layer(self, inplanes, outplanes):
        layer = []
        layer.append(nn.Conv2d(inplanes, outplanes, kernel_size = 1))
        layer.append(self._norm_layer(outplanes))
        layer.append(self.silu)
        return nn.Sequential(*layer)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.conv2(X)
        x3 = self.conv3(X)
        x4 = self.conv4(X)

        x12 = torch.add(x1, x2)
        x34 = torch.add(x3, x4)

        x1 = torch.add(self.atrous_conv1(x1),x12)
        x2 = torch.add(self.atrous_conv2(x2),x12)
        x3 = torch.add(self.atrous_conv3(x3),x34)
        x4 = torch.add(self.atrous_conv4(x4),x34)

        x12 = torch.cat([x1, x2], dim = 1)
        x34 = torch.cat([x3, x4], dim = 1)

        x12 = self.conv5(x12)
        x34 = self.conv5(x34)
        x = torch.cat([x12, x34], dim=1)
        x = self.convout(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        #assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        #assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        max =   input_tensor.view(batch_size, num_channels, -1).max(dim=2).values
        squeeze_tensor = mean + max
        # channel excitation
        fc_out_1 = F.gelu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class ConvMixer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.bn11 = nn.BatchNorm2d(self.dim)
        # self.bn12 = nn.BatchNorm2d(self.dim)
        # self.bn13 = nn.BatchNorm2d(self.dim)
        # self.cnn21 = nn.Conv2d(self.dim , self.dim , 3 , groups=self.dim , padding="same")
        # self.cnn22 = nn.Conv2d(self.dim , self.dim , 5 , groups=self.dim , padding="same")
        self.cnn23 = nn.Conv2d(self.dim , self.dim , 7 , groups=self.dim , padding="same")
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.cnn3 = nn.Conv2d(self.dim , self.dim , 1)

    def forward(self , x):
        # x = F.mish(self.bn11(self.cnn21(x)) + self.bn12(self.cnn22(x)) + self.bn13(self.cnn23(x))) + x #residual step and depthwise convolution
        x = F.mish(self.bn11(self.cnn23(x))) + x
        x = F.mish(self.bn2(self.cnn3(x)))                                      #pointwise convolution
        return x


class ConvTokenMix(nn.Module):
    def __init__(self, dim, norm_layer=None):
        super(ConvTokenMix, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(dim)
        self.convmixer = ConvMixer(dim)
        self.se = ChannelSELayer(dim)
        self.bn2 = norm_layer(dim)

    def forward(self, x):
        indentity = x
        x = self.convmixer(x)
        x = self.se(x) + x
        x = self.bn2(x)
        x += indentity
        x = F.mish(x)
        return x
    
class ConvBlock(nn.Module):
  def __init__(self, inchannel, out):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(inchannel, inchannel, kernel_size = 3, stride = 1, padding = 1,groups=inchannel),
        nn.Conv2d(inchannel, out, kernel_size = 1, stride = 1),
        nn.BatchNorm2d(out),
        nn.Mish(),
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out, out, kernel_size = 3, stride = 1, padding = 1,groups=out),
        nn.BatchNorm2d(out),
        nn.Mish(),
    )
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x
  

class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout=0.3):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.mish(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x
    
class DecoderBlock(nn.Module):
  def __init__(self, inchannels, outchannels, size):
    super().__init__()
    self.up = nn.Upsample(size = size)
    self.conv1 = nn.Sequential(
        nn.Conv2d(inchannels[0],inchannels[0],kernel_size=3,padding=1,groups=inchannels[0]),
        nn.Conv2d(inchannels[0],inchannels[0],1),
        nn.BatchNorm2d(inchannels[0]),
        nn.Mish(),
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(inchannels[1],inchannels[1],1),
        # nn.ConvTranspose2d(inchannels[1], inchannels[1], kernel_size=3, stride=2, padding=1, output_padding=1,groups=inchannels[1]),
        nn.Conv2d(inchannels[1], inchannels[1], kernel_size=3, padding=1,groups=inchannels[1]),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(inchannels[1],outchannels,1),
        nn.BatchNorm2d(outchannels),
        nn.Mish()
    )

  def forward(self, x, en = None, patch = None):
    if en is not None:
      x = x + en
      shortcut = x.clone()
      x = self.conv1(x)
      x = x + shortcut
    if en is not None:
      x = torch.cat([x,en], dim = 1 )
    x = self.conv2(x)
    return x
  

class TokenMixer(nn.Module):
    def __init__(self, num_features,num_patches, expansion_factor):
        super().__init__()
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(num_features)
        self.convtokenmix = ConvTokenMix(num_features)             #kernel_size = num_patches


    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1 = self.num_patches, p2 = self.num_patches )
        x = self.convtokenmix(x)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c', p1 = self.num_patches, p2 = self.num_patches )
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + residual
        return out


class MSConvmix(nn.Module):
    def __init__(self, num_features, num_patches, sqrt_num_patches, expansion_factor, dropout=0.3):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_features, sqrt_num_patches , expansion_factor
        )
        self.channel_mixer = ChannelMixer(
            num_features, num_patches, expansion_factor, dropout
        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
#     assert remainder == 0, "`image_size` must be divisibe by `patch_size`"''
    num_patches = sqrt_num_patches ** 2
    return sqrt_num_patches, num_patches

class MSConvmixModel(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=4,
        in_channels=3,
        num_features=64,
        expansion_factor=2,
        num_layers = [2,2,6,2],
        num_classes=2,
        dropout=0.3,
    ):
        self.sqrt_num_patches, self.num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        self.patch_size = patch_size

        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mixers0 = nn.Sequential(
            *[
                MSConvmix(num_features, self.num_patches,  self.sqrt_num_patches, expansion_factor, dropout,)
                for _ in range(num_layers[0])
            ]
        )
        self.patch_merge0 = PatchMerging([64, 64], 64)
        self.mixers1 = nn.Sequential(
            *[
                MSConvmix(num_features*2, self.num_patches//4,  self.sqrt_num_patches//2, expansion_factor, dropout,)
                for _ in range(num_layers[1])
            ]
        )
        self.patch_merge1 = PatchMerging([32, 32], 128)
        self.mixers2 = nn.Sequential(
            *[
                MSConvmix(num_features*4, self.num_patches//16,  self.sqrt_num_patches//4, expansion_factor, dropout,)
                for _ in range(num_layers[2])
            ]
        )
        self.patch_merge2 = PatchMerging([16, 16],256)
        self.mixers3 = nn.Sequential(
            *[
                MSConvmix(num_features*8, self.num_patches//32,  self.sqrt_num_patches//8, expansion_factor, dropout,)
                for _ in range(num_layers[3])
            ]
        )
        self.paspp = PASPP(512,512, output_stride = 4)
        self.conv_last = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(32,num_classes)
        )
        self.du1 = DecoderBlock([512, 512], 256, 16)
        self.du2 = DecoderBlock([256, 512], 128, 32)
        self.du3 = DecoderBlock([128, 256 ], 64, 64)
        self.du4 = DecoderBlock([64, 128 ], 32, 128)


    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _ , _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)

        embedding0 = self.mixers0(patches)
        embedding1 = self.patch_merge0(embedding0)

        embedding1 = self.mixers1(embedding1)
        embedding2 = self.patch_merge1(embedding1)

        embedding2 = self.mixers2(embedding2)
        embedding3 = self.patch_merge2(embedding2)

        embedding = self.mixers3(embedding3)

        embedding = rearrange(embedding, 'b (p1 p2) c -> b c p1 p2', p1 = self.sqrt_num_patches//8, p2 = self.sqrt_num_patches//8 )
        embedding = self.paspp(embedding)

        embedding2 = rearrange(embedding2, 'b (p1 p2) c -> b c p1 p2', p1 = self.sqrt_num_patches//4, p2 = self.sqrt_num_patches//4 )
        embedding1 = rearrange(embedding1, 'b (p1 p2) c -> b c p1 p2', p1 = self.sqrt_num_patches//2, p2 = self.sqrt_num_patches//2 )
        embedding0 = rearrange(embedding0, 'b (p1 p2) c -> b c p1 p2', p1 = self.sqrt_num_patches, p2 = self.sqrt_num_patches )

        x = self.du1(embedding)
        x = self.du2(x, embedding2, 16)

        x = self.du3(x, embedding1, 32)
        x = self.du4(x, embedding0, 64)

        x= self.conv_last(x)

        return  x