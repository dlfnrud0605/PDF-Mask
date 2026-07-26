"""
CLIP (OpenAI MIT License) ModifiedResNet을 그대로 포팅.
MMOCR의 CLIPResNet과 동일한 키 구조 (backbone.stem.*, backbone.layer*.*)를 유지함.
"""
import torch.nn as nn
import torch.nn.functional as F


class CLIPBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 다운샘플링을 stride conv 대신 AvgPool로 처리 (CLIP 방식)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or inplanes != planes * 4:
            # MMOCR 체크포인트 키: downsample.0(pool), downsample.1(conv), downsample.2(bn)
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(inplanes, planes * 4, 1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class CLIPResNet(nn.Module):
    """
    CLIP ModifiedResNet (ResNet-50 설정).
    - stem: 3 × 3×3 conv + AvgPool (표준 ResNet의 7×7 + MaxPool 대체)
    - Bottleneck: AvgPool 다운샘플링 (stride conv 없음)
    - DCN 미사용
    출력: (C2, C3, C4, C5) 채널 = (256, 512, 1024, 2048)
    """
    def __init__(self, layers=(3, 4, 6, 3), width=64):
        super().__init__()
        self._inplanes = width

        # MMOCR 키: stem.0, stem.1, stem.3, stem.4, stem.6, stem.7 (인덱스 2,5,8=ReLU, 9=AvgPool 파라미터 없음)
        self.stem = nn.Sequential(
            nn.Conv2d(3, width // 2, 3, stride=2, padding=1, bias=False),  # 0
            nn.BatchNorm2d(width // 2),                                     # 1
            nn.ReLU(inplace=True),                                          # 2
            nn.Conv2d(width // 2, width // 2, 3, padding=1, bias=False),   # 3
            nn.BatchNorm2d(width // 2),                                     # 4
            nn.ReLU(inplace=True),                                          # 5
            nn.Conv2d(width // 2, width, 3, padding=1, bias=False),        # 6
            nn.BatchNorm2d(width),                                          # 7
            nn.ReLU(inplace=True),                                          # 8
            nn.AvgPool2d(2),                                                # 9
        )

        self.layer1 = self._make_layer(width,     layers[0], stride=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

    def _make_layer(self, planes, num_blocks, stride=1):
        layers = [CLIPBottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * 4
        for _ in range(1, num_blocks):
            layers.append(CLIPBottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)   # 1/4, ch=256
        c3 = self.layer2(c2)  # 1/8, ch=512
        c4 = self.layer3(c3)  # 1/16, ch=1024
        c5 = self.layer4(c4)  # 1/32, ch=2048
        return c2, c3, c4, c5
