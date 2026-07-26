import torch
import numpy as np
import cv2
import os
import pyclipper
from shapely.geometry import Polygon
from typing import List

from app.core.base import BaseModel
from .libs.dbnet.model import DBNetPP


class DBNetPPDetector(BaseModel):
    """
    DBNet++ oCLIP Detector.
    MMOCR dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015 가중치와 호환.
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model_path = self.config.get("path", "models/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.pth")
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.box_thresh = self.config.get("box_thresh", 0.3)
        self.binarize_thresh = self.config.get("binarize_thresh", 0.3)
        self.unclip_ratio = self.config.get("unclip_ratio", 1.5)
        self.short_size = self.config.get("short_size", 736)
        self.max_size = self.config.get("max_size", 2560)
        self.return_heatmap = self.config.get("return_heatmap", False)

        self.model = DBNetPP()
        if os.path.exists(self.model_path):
            self._load_weights()
        else:
            print(f"[DBNet++] 경고: 가중치 파일 없음 → {self.model_path}")

        self.model.to(self.device).eval()

    def _load_weights(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # MMOCR MMEngine 체크포인트 형식 처리
        sd = ckpt.get("state_dict", ckpt)

        result = self.model.load_state_dict(sd, strict=False)
        print(f"[DBNet++] 로드 완료: {self.model_path}")
        if result.missing_keys:
            missing = result.missing_keys
            print(f"[DBNet++] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if result.unexpected_keys:
            unexp = result.unexpected_keys
            print(f"[DBNet++] Unexpected keys ({len(unexp)}): {unexp[:5]}{'...' if len(unexp) > 5 else ''}")

    def preprocess(self, image_np):
        h, w = image_np.shape[:2]
        # 학습과 동일하게 긴 변 기준 리사이즈 (scale=(1280,1280), keep_ratio=True)
        scale = self.short_size / max(h, w) if max(h, w) > 0 else 1.0
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = new_h if new_h % 32 == 0 else (new_h // 32 + 1) * 32
        new_w = new_w if new_w % 32 == 0 else (new_w // 32 + 1) * 32

        img = cv2.resize(image_np, (new_w, new_h)).astype(np.float32)
        img = img[:, :, ::-1]  # BGR → RGB (MMOCR 학습과 동일한 채널 순서)
        img /= 255.0
        img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)

        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return tensor, (h, w), (new_h, new_w)

    def infer(self, image_np: np.ndarray):
        if image_np is None or image_np.size == 0:
            return ([], None) if self.return_heatmap else []

        img_tensor, orig_size, proc_size = self.preprocess(image_np)

        with torch.no_grad():
            binary, _ = self.model(img_tensor)

        prob_map = binary[0, 0].cpu().numpy()
        boxes = self._post_process(prob_map, orig_size, proc_size)

        results = [{"word_box": b, "char_boxes": [b]} for b in boxes]
        if self.return_heatmap:
            return results, prob_map
        return results

    def _post_process(self, prob_map, orig_size, proc_size):
        h_orig, w_orig = orig_size
        h_proc, w_proc = proc_size

        seg = (prob_map > self.binarize_thresh).astype(np.uint8)
        # [수정] MMOCR 가중치는 파편화가 적으므로, 단어 단위 분리를 위해 강제 팽창(dilate)을 제거합니다.
        
        contours, _ = cv2.findContours(
            (seg * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            # 최소 면적 필터: 노이즈 blob 제거 (MMOCR DBPostprocessor 기준)
            if cv2.contourArea(contour) < 16:
                continue
            score = self._score(prob_map, seg, contour)
            if score < self.box_thresh:
                continue

            pts = contour[:, 0, :]
            poly = Polygon(pts)
            if poly.length <= 0:
                continue

            distance = poly.area * self.unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if not expanded:
                continue

            exp = np.array(expanded[0])
            x1, y1 = np.min(exp, axis=0)
            x2, y2 = np.max(exp, axis=0)
            x1 = int(x1 * w_orig / w_proc)
            y1 = int(y1 * h_orig / h_proc)
            x2 = int(x2 * w_orig / w_proc)
            y2 = int(y2 * h_orig / h_proc)
            boxes.append([max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)])

        return boxes

    def _score(self, prob_map, seg, contour):
        mask = np.zeros(prob_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        valid = (mask == 1) & (seg == 1)
        return float(np.mean(prob_map[valid])) if np.any(valid) else 0.0
