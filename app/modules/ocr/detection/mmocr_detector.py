import numpy as np
from typing import Any

from app.core.base import BaseModel


class MMOCRTextDetector(BaseModel):
    """
    MMOCR TextDetInferencer 기반 텍스트 탐지기.
    학습 평가와 동일한 DBPostprocessor를 사용하므로 커스텀 후처리보다 정확함.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model_config  = self.config.get("config", "configs/dbnetpp_resnet50_custom.py")
        self.weights       = self.config.get("weights", "")
        self.device        = self.config.get("device", "cuda")
        self.return_heatmap = self.config.get("return_heatmap", False)

        from mmocr.apis import TextDetInferencer
        import signal
        
        # MMEngine이 스레드 내부에서 signal을 등록하려다 터지는 것을 방지 (Monkey-patching)
        original_signal = signal.signal
        def noop_signal(*args, **kwargs):
            pass
        signal.signal = noop_signal
        
        try:
            print(f"[MMOCRTextDetector] 모델 로딩: {self.weights}")
            self.inferencer = TextDetInferencer(
                model=self.model_config, weights=self.weights, device=self.device
            )
        finally:
            signal.signal = original_signal

    def infer_with_heatmap(self, image_np: np.ndarray):
        """word 결과 + DBNet++ prob_map(히트맵) 함께 반환."""
        captured = {}

        def _hook(module, inp, output):
            if isinstance(output, dict) and 'prob_map' in output:
                captured['prob_map'] = output['prob_map'].detach().cpu()
            elif hasattr(output, 'detach'):
                captured['prob_map'] = output.detach().cpu()

        handle = self.inferencer.model.det_head.register_forward_hook(_hook)
        try:
            results = self.infer(image_np)
        finally:
            handle.remove()

        prob_map = None
        if 'prob_map' in captured:
            pm = captured['prob_map']
            if pm.ndim == 4: pm = pm[0, 0]
            elif pm.ndim == 3: pm = pm[0]
            prob_map = pm.numpy()

        return results, prob_map

    def infer(self, image_np: np.ndarray) -> Any:
        if image_np is None or image_np.size == 0:
            return ([], None) if self.return_heatmap else []

        # DBGGNetPP는 RGB로 학습됨 — BGR 입력이면 채널 반전
        img_rgb = image_np[:, :, ::-1].copy()

        import signal
        original_signal = signal.signal
        def noop_signal(*args, **kwargs):
            pass
        signal.signal = noop_signal

        try:
            result = self.inferencer(img_rgb, return_datasamples=True)
        finally:
            signal.signal = original_signal

        datasample = result["predictions"][0]
        polygons = datasample.pred_instances.polygons

        boxes = []
        for poly in polygons:
            pts = np.array(poly).reshape(-1, 2)
            x1, y1 = pts.min(axis=0).astype(int)
            x2, y2 = pts.max(axis=0).astype(int)
            boxes.append([x1, y1, x2, y2])

        results = [{"word_box": b, "char_boxes": [b]} for b in boxes]

        if self.return_heatmap:
            return results, None  # TextDetInferencer는 prob_map을 외부로 노출하지 않음
        return results
