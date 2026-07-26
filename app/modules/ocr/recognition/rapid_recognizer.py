import os
import cv2
import numpy as np
from typing import List, Any

from app.core.base import BaseModel


class RapidRecognizer(BaseModel):
    """
    RapidOCR(ONNX Runtime) 기반 텍스트 인식 모듈.
    PaddleOCR 한국어 모델을 ONNX로 실행 — PyTorch와 같은 프로세스에서 충돌 없이 동작.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}

        self.drop_threshold = config.get("drop_threshold", 0.6)

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "rapidocr_onnxruntime가 설치되어 있지 않습니다.\n"
                "pip install rapidocr_onnxruntime"
            )

        self.ocr = RapidOCR()
        print("[RapidRecognizer] 모델 로딩 완료.")

    def infer(self, image_or_images) -> List[Any]:
        """
        단일 NumPy 이미지 또는 이미지 리스트를 받아 인식 결과를 반환.
        - 단일 입력: List[Dict]  예) [{"text": "안녕", "conf": 0.97}]
        - 배치 입력: List[List[Dict]]
        """
        is_batch = isinstance(image_or_images, list)
        images = image_or_images if is_batch else [image_or_images]

        final_results = []
        for img in images:
            if img is None or img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5:
                final_results.append([])
                continue

            try:
                result, _ = self.ocr(img, use_det=False, use_cls=False, use_rec=True)
                texts = []
                if result:
                    for item in result:
                        text, conf = item[0], float(item[1])
                        if conf >= self.drop_threshold:
                            texts.append({"text": str(text), "conf": conf})
                final_results.append(texts)
            except Exception as e:
                print(f"[RapidRecognizer 에러] {e}")
                final_results.append([])

        return final_results if is_batch else (final_results[0] if final_results else [])
