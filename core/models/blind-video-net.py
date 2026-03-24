import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import register_model
from core.models import register_unet

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x


class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x
    
class crop2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-2, 0:W]
        return x

class shift2(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,2,0))
        self.crop = crop2()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        if self.blind:
            x = self.crop(x)
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') #changed mode from 'nearest' or 'bilinear'
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class BlurPool(nn.Module):
    """
    Anti-aliased downsampling with causal constraint: MaxPool2d(stride=1) -> 3x3 Gaussian blur(stride=2).
    Adapted from Zhang 2019 / Hoeck et al. 2022 (N2V2).
      
    Two shifts required to maintain causality:
      shift_pre:  standard causal offset before maxpool
      shift_post: compensates for MaxPool(stride=1) reaching +1 position
    
    Causal blur kernel (zeroed top row) prevents upward information flow.
    """

    def __init__(self, channels, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_pre = shift()
            self.shift_post = shift()

        self.pad_max = nn.ReplicationPad2d((0, 1, 0, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.blur = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1,
            groups=channels, bias=False
        )
        if blind:
            blur_kernel = torch.tensor(
                [[0, 0, 0],
                 [2, 4, 2],
                 [1, 2, 1]], dtype=torch.float32
            ) / 12.0
        else:
            blur_kernel = torch.tensor(
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]], dtype=torch.float32
            ) / 16.0

        with torch.no_grad():
            self.blur.weight.copy_(
                blur_kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3)
            )
        self.blur.weight.requires_grad = False

    def forward(self, x):
        if self.blind:
            x = self.shift_pre(x)
        x = self.pad_max(x)
        x = self.maxpool(x)
        if self.blind:
            x = self.shift_post(x)
        x = self.blur(x)
        return x

class AvgBlurPool(nn.Module):
    """No MaxPool at all. Just blur + stride-2 subsample."""
    def __init__(self, channels, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()

        # Causal average: only look at current and below
        self.blur = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1,
            groups=channels, bias=False
        )
        if blind:
            blur_kernel = torch.tensor(
                [[0, 0, 0],
                 [2, 4, 2],
                 [1, 2, 1]], dtype=torch.float32
            ) / 12.0
        else:
            blur_kernel = torch.tensor(
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]], dtype=torch.float32
            ) / 16.0

        with torch.no_grad():
            self.blur.weight.copy_(
                blur_kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3)
            )
        self.blur.weight.requires_grad = False

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.blur(x)
        return x

class SPD_Pool(nn.Module):
    """
    Space-to-Depth downsampling. No information loss, no aliasing.
    Rearranges 2×2 spatial blocks into 4× channels.
    """
    def __init__(self, channels, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        # After SPD: 4*channels input → channels output
        self.conv = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        # Space-to-depth: (B, C, H, W) → (B, 4C, H/2, W/2)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)  # (B, C, 2, 2, H/2, W/2)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        # Reduce channels back
        x = self.conv(x)
        return x

class ENC_Conv_Blur(nn.Module):
    """Encoder block with BlurPool instead of MaxPool. Otherwise identical to ENC_Conv."""
    def __init__(self, in_channels, mid_channels, out_channels,
                 bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = BlurPool(out_channels, blind=blind) # Manually change here class for Pool variants (BlurPool, AvgPool, SPD_Blur, etc.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x


class DEC_Conv_NoSkip(nn.Module):
    """
    Decoder block WITHOUT skip connection (N2V2: remove top-level skip).
    Upsamples and processes, but does NOT concatenate the encoder features.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_for_size):
        """x_for_size is only used for spatial dimensions, NOT concatenated."""
        x = self.upsample(x)
        diffY = x_for_size.size()[2] - x.size()[2]
        diffX = x_for_size.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        # NO torch.cat — this is the key N2V2 difference
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


@register_unet("original")
class Blind_UNet(nn.Module):
    """2-level, MaxPool. Bottleneck at input/4."""
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96+n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

@register_unet("blur")
class Blind_UNet_Blur(nn.Module):
    """
    2-level U-Net with BlurPool. Same depth as original, all skips retained.
    Bottleneck at input/4.
    """
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv_Blur(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv_Blur(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv_Blur(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96 + n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

@register_unet("deep")
class Blind_UNet_Deep(nn.Module):
    """3-level, MaxPool. Bottleneck at input/8. """
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc4 = ENC_Conv(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec3 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec2 = DEC_Conv(96+48, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96+n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)     # H/2, 48ch
        x2 = self.enc2(x1)        # H/4, 48ch
        x3 = self.enc3(x2)        # H/8, 48ch
        x  = self.enc4(x3)        # H/8, 48ch (bottleneck)
        x = self.dec3(x, x2)      # -> H/4
        x = self.dec2(x, x1)      # -> H/2
        x = self.dec1(x, input)    # -> H
        return x


@register_unet("deep_blur")
class Blind_UNet_Deep_Blur(nn.Module):
    """3-level, BlurPool (causal). Bottleneck at input/8."""
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv_Blur(n_channels, 48, 48, bias=bias, blind=blind)       # -> H/2
        self.enc2 = ENC_Conv_Blur(48, 48, 48, bias=bias, blind=blind)               # -> H/4
        self.enc3 = ENC_Conv_Blur(48, 48, 48, bias=bias, blind=blind)               # -> H/8
        self.enc4 = ENC_Conv_Blur(48, 96, 48, bias=bias, reduce=False, blind=blind) # bottleneck
        self.dec3 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)                    # skip from x2
        self.dec2 = DEC_Conv(96 + 48, 96, 96, bias=bias, blind=blind)               # skip from x1
        self.dec1 = DEC_Conv(96 + n_channels, 96, n_output, bias=bias, blind=blind) # skip from input

    def forward(self, input):
        x1 = self.enc1(input)     # H/2,  48ch
        x2 = self.enc2(x1)        # H/4,  48ch
        x3 = self.enc3(x2)        # H/8,  48ch
        x  = self.enc4(x3)        # H/8,  48ch (bottleneck)
        x = self.dec3(x, x2)      # -> H/4,  cat with x2 (48ch)  -> 96 in
        x = self.dec2(x, x1)      # -> H/2,  cat with x1 (48ch)  -> 144 in
        x = self.dec1(x, input)    # -> H,    cat with input      -> 96+n_ch in
        return x


# ============================================================================
# New 4D models (UNet variants can be injected)
# ============================================================================

@register_model("blind-video-net-4d-cross")
class BlindVideoNet4DCross(nn.Module):
    """
    9-channel model with 4 directional groups (V, H, diag1, diag2).
    24 features per group × 4 = 96 total. Same feature budget as cross model.
    
    Input: (batch, 9, Qx, Qy) — full 3×3 grid, center at index 4
    Output: (batch, 1, Qx, Qy)
    """
    def __init__(self, channels_per_frame=1, out_channels=1, bias=False,
                 blind=True, sigma_known=True, unet_cls=None):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        
        if unet_cls is None:
            unet_cls = Blind_UNet
        
        # 4 groups × 24 features = 96
        self.denoiser_1 = unet_cls(n_channels=3*channels_per_frame, n_output=24,
                                    bias=bias, blind=blind)
        self.denoiser_2 = unet_cls(n_channels=96, n_output=96,
                                    bias=bias, blind=blind)
        
        if not sigma_known:
            self.sigma_net = unet_cls(n_channels=9*channels_per_frame, n_output=1,
                                      bias=False, blind=False)
        if blind:
            self.shift = shift()
        
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=1)
        parser.add_argument("--out-channels", type=int, default=1)
        parser.add_argument("--bias", action='store_true')
        parser.add_argument("--normal", action='store_true')
        parser.add_argument("--blind-noise", action='store_true')

    @classmethod
    def build_model(cls, args, unet_cls=None):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels,
                   bias=args.bias, blind=(not args.normal),
                   sigma_known=(not args.blind_noise), unet_cls=unet_cls)

    def forward(self, x):
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))

        if H > W:
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode='reflect')
        elif W > H:
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode='reflect')

        # 4 groups through center, each a line in the 3×3 grid:
        #   0 1 2
        #   3 4 5
        #   6 7 8
        i1 = self.rotate(x[:, [0, 4, 8], :, :])  # diagonal ↘
        i2 = self.rotate(x[:, [1, 4, 7], :, :])  # vertical ↓
        i3 = self.rotate(x[:, [2, 4, 6], :, :])  # diagonal ↙
        i4 = self.rotate(x[:, [3, 4, 5], :, :])  # horizontal →

        y1 = self.denoiser_1(i1)  # each → 24 features
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)
        y4 = self.denoiser_1(i4)

        y = torch.cat((y1, y2, y3, y4), dim=1)  # 4 × 24 = 96
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        x = F.relu(x)

        if H > W:
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif W > H:
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x

@register_model("blind-video-net-5d-cross")
class BlindVideoNet5DCross(nn.Module):
    """
    9-channel model with 4 directional groups and shift2 (2px blind spot).
    24 features per group x 4 = 96 total.

    Input: (batch, 9, Qx, Qy) — full 3x3 grid, center at index 4
    Output: (batch, 1, Qx, Qy)

    Groups (each a line through center):
        [0, 4, 8] — diagonal
        [1, 4, 7] — vertical / extended temporal
        [2, 4, 6] — anti-diagonal
        [3, 4, 5] — horizontal / direct temporal

    Use with is_include_neighbor=False (shift2 blinds center + immediate neighbors).
    """
    def __init__(self, channels_per_frame=1, out_channels=1, bias=False,
                 blind=True, sigma_known=True, unet_cls=None):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()

        if unet_cls is None:
            unet_cls = Blind_UNet

        # 4 groups x 24 features = 96
        self.denoiser_1 = unet_cls(n_channels=3*channels_per_frame, n_output=24,
                                    bias=bias, blind=blind)
        self.denoiser_2 = unet_cls(n_channels=96, n_output=96,
                                    bias=bias, blind=blind)

        if not sigma_known:
            self.sigma_net = unet_cls(n_channels=9*channels_per_frame, n_output=1,
                                      bias=False, blind=False)
        if blind:
            self.shift = shift2()  # <-- 2px blind spot (only difference from 4d-cross)

        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=1)
        parser.add_argument("--out-channels", type=int, default=1)
        parser.add_argument("--bias", action='store_true')
        parser.add_argument("--normal", action='store_true')
        parser.add_argument("--blind-noise", action='store_true')

    @classmethod
    def build_model(cls, args, unet_cls=None):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels,
                   bias=args.bias, blind=(not args.normal),
                   sigma_known=(not args.blind_noise), unet_cls=unet_cls)

    def forward(self, x):
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))

        if H > W:
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode='reflect')
        elif W > H:
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode='reflect')

        # 4 groups through center, each a line in the 3x3 grid:
        #   0 1 2
        #   3 4 5
        #   6 7 8
        i1 = self.rotate(x[:, [0, 4, 8], :, :])  # diagonal
        i2 = self.rotate(x[:, [1, 4, 7], :, :])  # vertical
        i3 = self.rotate(x[:, [2, 4, 6], :, :])  # anti-diagonal
        i4 = self.rotate(x[:, [3, 4, 5], :, :])  # horizontal

        y1 = self.denoiser_1(i1)  # each -> 24 features
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)
        y4 = self.denoiser_1(i4)

        y = torch.cat((y1, y2, y3, y4), dim=1)  # 4 x 24 = 96
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        x = F.relu(x)

        if H > W:
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif W > H:
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x






### LEGACY MODELS

@register_model("blind-spot-net-4")
class BlindSpotNet(nn.Module):
    def __init__(self, n_channels=3, n_output=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.unet = _UNet(n_channels=n_channels, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = _UNet(n_channels=n_channels, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--in-channels", type=int, default=3, help="number of input channels")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(n_channels=args.in_channels, n_output=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.unet(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma

@register_model("blind-video-net-d1-4")
class BlindVideoNetD1(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = _UNet(n_channels=3*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.denoiser_1(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma
      
@register_model("blind-video-net-4")
class BlindVideoNet(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=32, bias=bias, blind=blind)
        self.denoiser_2 = _UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = _UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i1 = self.rotate(x[:, 0:(3*self.c), :, :])
        i2 = self.rotate(x[:, self.c:(4*self.c), :, :])
        i3 = self.rotate(x[:, (2*self.c):(5*self.c), :, :])

        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)

        y = torch.cat((y1, y2, y3), dim=1)
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x

@register_model("blind-video-net-4-4d")
class BlindVideoNet4D(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        
        # First stage denoiser for triplets of frames - maintained at 32 features per output
        # But now processing 4 groups instead of 3, we reduce to 24 features per group to keep total at 96
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=24, bias=bias, blind=blind)
        
        # Second stage denoiser with same input/output channels as original
        self.denoiser_2 = _UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        
        if not sigma_known:
            # Sigma estimation network for 9 input frames
            self.sigma_net = _UNet(n_channels=9*channels_per_frame, n_output=1, bias=False, blind=False)
            
        if blind:
            self.shift = shift()
            
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        # Process four groups of frames in a 3x3 grid:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Where 4 is the center frame
        
        # Group 1: Diagonal from top-left to bottom-right (0, 4, 8)
        i1 = self.rotate(x[:, [0, 4, 8], :, :])
        
        # Group 2: Vertical line (1, 4, 7)
        i2 = self.rotate(x[:, [1, 4, 7], :, :])
        
        # Group 3: Diagonal from top-right to bottom-left (2, 4, 6)
        i3 = self.rotate(x[:, [2, 4, 6], :, :])
        
        # Group 4: Horizontal line (3, 4, 5)
        i4 = self.rotate(x[:, [3, 4, 5], :, :])

        # Process each group through the first denoiser
        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)
        y4 = self.denoiser_1(i4)

        # Concatenate all outputs from the first stage
        y = torch.cat((y1, y2, y3, y4), dim=1)
        
        # Process through the second denoiser
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x

@register_model("blind-video-net-5")
class BlindVideoNet(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=32, bias=bias, blind=blind)
        self.denoiser_2 = _UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = _UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift2()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i1 = self.rotate(x[:, 0:(3*self.c), :, :])
        i2 = self.rotate(x[:, self.c:(4*self.c), :, :])
        i3 = self.rotate(x[:, (2*self.c):(5*self.c), :, :])

        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)

        y = torch.cat((y1, y2, y3), dim=1)
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x
    
@register_model("blind-video-net-5-4d")
class BlindVideoNet4D(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        
        # First stage denoiser for triplets of frames - maintained at 32 features per output
        # But now processing 4 groups instead of 3, we reduce to 24 features per group to keep total at 96
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=24, bias=bias, blind=blind)
        
        # Second stage denoiser with same input/output channels as original
        self.denoiser_2 = _UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        
        if not sigma_known:
            # Sigma estimation network for 9 input frames
            self.sigma_net = _UNet(n_channels=9*channels_per_frame, n_output=1, bias=False, blind=False)
            
        if blind:
            self.shift = shift2()
            
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        # Process four groups of frames in a 3x3 grid:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Where 4 is the center frame
        
        # Group 1: Diagonal from top-left to bottom-right (0, 4, 8)
        i1 = self.rotate(x[:, [0, 4, 8], :, :])
        
        # Group 2: Vertical line (1, 4, 7)
        i2 = self.rotate(x[:, [1, 4, 7], :, :])
        
        # Group 3: Diagonal from top-right to bottom-left (2, 4, 6)
        i3 = self.rotate(x[:, [2, 4, 6], :, :])
        
        # Group 4: Horizontal line (3, 4, 5)
        i4 = self.rotate(x[:, [3, 4, 5], :, :])

        # Process each group through the first denoiser
        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)
        y4 = self.denoiser_1(i4)

        # Concatenate all outputs from the first stage
        y = torch.cat((y1, y2, y3, y4), dim=1)
        
        # Process through the second denoiser
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        x = F.relu(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x

# ============================================================================
# Cross-geometry models for 4D-STEM (5 channels, V/H decomposition)
#
# Input: 5 channels in cross order [top, left, center, right, bottom]
#   center is at index 2
#
# Decomposition:
#   Group 1 (vertical):   channels [0, 2, 4]  =  top,  center, bottom
#   Group 2 (horizontal): channels [1, 2, 3]  =  left, center, right
#
# 2 groups x 48 features = 96 total (same feature budget as the 3-group models)
# ============================================================================

@register_model("blind-video-net-4-cross")
class BlindVideoNetCross(nn.Module):
    """Cross-geometry, shift (1-pixel blind spot). Use with is_include_neighbor=True."""
    def __init__(self, channels_per_frame=1, out_channels=1, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=48, bias=bias, blind=blind)
        self.denoiser_2 = _UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = _UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=1, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=1, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        # Cross decomposition
        # Vertical:   [top, center, bottom] = channels [0, 2, 4]
        # Horizontal: [left, center, right] = channels [1, 2, 3]
        i_vert = self.rotate(x[:, [0, 2, 4], :, :])
        i_horz = self.rotate(x[:, [1, 2, 3], :, :])

        y_vert = self.denoiser_1(i_vert)
        y_horz = self.denoiser_1(i_horz)

        y = torch.cat((y_vert, y_horz), dim=1)  # 48 + 48 = 96
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        x = F.relu(x)

        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x


@register_model("blind-video-net-5-cross")
class BlindVideoNetCrossWide(nn.Module):
    """Cross-geometry, shift2 (2-pixel blind spot). Use with is_include_neighbor=False."""
    def __init__(self, channels_per_frame=1, out_channels=1, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = _UNet(n_channels=3*channels_per_frame, n_output=48, bias=bias, blind=blind)
        self.denoiser_2 = _UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = _UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift2()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=1, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=1, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i_vert = self.rotate(x[:, [0, 2, 4], :, :])
        i_horz = self.rotate(x[:, [1, 2, 3], :, :])

        y_vert = self.denoiser_1(i_vert)
        y_horz = self.denoiser_1(i_horz)

        y = torch.cat((y_vert, y_horz), dim=1)
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        x = F.relu(x)

        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x
