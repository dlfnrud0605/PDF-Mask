"""
MMOCR FPNC neck + ASFAttn + DBHead.
MMOCR 체크포인트 키 구조에 맞춰 설계:
  neck.lateral_convs.{i}.conv.*
  neck.smooth_convs.{i}.conv.*
  neck.asf_conv.conv.*
  neck.asf_attn.channel_wise.{i}.conv.*
  neck.asf_attn.spatial_wise.{i}.conv.*
  neck.asf_attn.attention_wise.conv.*
  det_head.binarize.*
  det_head.threshold.*
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """MMOCR ConvModule 호환 래퍼: conv 속성으로 접근 가능."""
    def __init__(self, in_ch, out_ch, kernel, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ASFAttn(nn.Module):
    """Adaptive Scale Fusion attention (MMOCR ScaleChannelSpatial 방식)."""
    def __init__(self, in_channels, num_scales=4):
        super().__init__()
        mid = in_channels // 4
        self.channel_wise = nn.ModuleList([
            ConvLayer(in_channels, mid, 1),
            ConvLayer(mid, in_channels, 1),
        ])
        self.spatial_wise = nn.ModuleList([
            ConvLayer(1, 1, 3, padding=1),
            ConvLayer(1, 1, 1),
        ])
        self.attention_wise = ConvLayer(in_channels, num_scales, 1)

    def forward(self, x):
        # 채널 어텐션
        ch = F.adaptive_avg_pool2d(x, 1)
        ch = F.relu(self.channel_wise[0].conv(ch), inplace=True)
        ch = self.channel_wise[1].conv(ch).sigmoid()
        x = x * ch + x

        # 공간 어텐션
        sp = x.mean(dim=1, keepdim=True)
        sp = F.relu(self.spatial_wise[0].conv(sp), inplace=True)
        sp = self.spatial_wise[1].conv(sp).sigmoid()
        x = x * sp

        # 스케일별 가중치 (num_scales개)
        return self.attention_wise.conv(x).sigmoid()


class FPNC(nn.Module):
    """
    Feature Pyramid Network with Concatenation.
    in_channels: 백본 각 스테이지 출력 채널 [C2, C3, C4, C5]
    lateral_channels: lateral 1×1 conv 출력 채널
    out_channels: smooth 3×3 conv 출력 채널 (스케일당)
    """
    def __init__(self, in_channels=(256, 512, 1024, 2048),
                 lateral_channels=256, out_channels=64):
        super().__init__()
        n = len(in_channels)
        self.lateral_convs = nn.ModuleList([
            ConvLayer(c, lateral_channels, 1) for c in in_channels
        ])
        self.smooth_convs = nn.ModuleList([
            ConvLayer(lateral_channels, out_channels, 3, padding=1) for _ in range(n)
        ])
        total_ch = out_channels * n  # 64 * 4 = 256
        self.asf_conv = ConvLayer(total_ch, total_ch, 3, padding=1, bias=True)
        self.asf_attn = ASFAttn(total_ch, num_scales=n)

    def forward(self, features):
        c2, c3, c4, c5 = features

        # Lateral connections
        lat = [conv(features[i]) for i, conv in enumerate(self.lateral_convs)]

        # Top-down FPN 융합
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(
                lat[i], size=lat[i - 1].shape[2:], mode='nearest')

        # Smooth convs
        smooth = [conv(lat[i]) for i, conv in enumerate(self.smooth_convs)]

        # 모두 C2 해상도로 업샘플
        h, w = smooth[0].shape[2:]
        for i in range(1, len(smooth)):
            smooth[i] = F.interpolate(smooth[i], size=(h, w), mode='nearest')

        # 연결 후 ASF 어텐션
        fuse = torch.cat(smooth, dim=1)
        asf_feat = self.asf_conv.conv(fuse)
        scale_weights = self.asf_attn(asf_feat)  # (B, 4, H, W)

        # 스케일별 가중치 적용 후 재연결
        out = torch.cat(
            [scale_weights[:, i:i+1] * smooth[i] for i in range(len(smooth))],
            dim=1
        )
        return out  # (B, 256, H, W)


class DBHead(nn.Module):
    """
    DBNet++ 이진화 헤드.
    in_channels=256 → ConvTranspose 4× 업샘플 → 1채널 확률 맵
    MMOCR 체크포인트 키: det_head.binarize.*, det_head.threshold.*
    """
    def __init__(self, in_channels=256):
        super().__init__()
        mid = in_channels // 4  # 64

        def _branch():
            return nn.Sequential(
                nn.Conv2d(in_channels, mid, 3, bias=False, padding=1),  # [0]
                nn.BatchNorm2d(mid),                                     # [1]
                nn.ReLU(inplace=True),                                   # [2]
                nn.ConvTranspose2d(mid, mid, 2, 2),                      # [3]
                nn.BatchNorm2d(mid),                                     # [4]
                nn.ReLU(inplace=True),                                   # [5]
                nn.ConvTranspose2d(mid, 1, 2, 2),                        # [6]
                nn.Sigmoid(),                                            # [7]
            )

        self.binarize = _branch()
        self.threshold = _branch()

    def forward(self, x):
        prob = self.binarize(x)
        thresh = self.threshold(x)
        return prob, thresh
