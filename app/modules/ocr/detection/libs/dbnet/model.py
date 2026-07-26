"""
DBNet++ oCLIP 모델 조립기.
MMOCR dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015 체크포인트와 키가 1:1 호환.
"""
import torch.nn as nn
from .clip_backbone import CLIPResNet
from .fpnc import FPNC, DBHead


class DBNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPResNet(layers=(3, 4, 6, 3), width=64)
        self.neck = FPNC(
            in_channels=(256, 512, 1024, 2048),
            lateral_channels=256,
            out_channels=64,
        )
        self.det_head = DBHead(in_channels=256)

    def forward(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)
        prob, thresh = self.det_head(neck_out)
        return prob, thresh
