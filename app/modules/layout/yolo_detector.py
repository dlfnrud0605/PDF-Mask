try:
    from doclayout_yolo import YOLOv10 as YOLO
except ImportError:
    from ultralytics import YOLO
from app.core.base import BaseModel
from typing import Any, List, Dict
import numpy as np
import os

class YOLODetector(BaseModel):
    """
    YOLOv8 모델을 감싸는(Wrapper) 클래스입니다.
    BaseModel의 규격을 따르기 때문에, 외부에서는 이 클래스가 YOLO인지 뭔지 몰라도
    그냥 infer() 함수만 부르면 똑같이 좌표를 얻을 수 있습니다.
    """
    def __init__(self, config: dict | None = None):
        # 1. 모델 가중치 파일 경로 설정 (기본은 가벼운 yolov8n으로 설정)
        model_path = config.get("path", "models/yolov8n.pt") if config else "models/yolov8n.pt"
        
        # [안전장치] 만약 config에 적힌 경로에 실제 파일이 없다면?
        if model_path != "models/yolov8n.pt" and not os.path.exists(model_path):
            print(f"[경고] {model_path} 경로에 모델 파일이 없습니다! 기본 모델(models/yolov8n.pt)로 비상 가동합니다.")
            model_path = "models/yolov8n.pt"

        # 2. 실제 YOLO 모델 메모리에 로드 (만약 파일이 없으면 자동으로 ultralytics가 인터넷에서 받음)
        self.model = YOLO(model_path)
        
        # 3. 사용할 장치 및 임계값 설정
        self.device = config.get("device", "cuda") if config else "cuda"
        self.conf_thresh = config.get("conf_thresh", 0.15) if config else 0.15
        self.iou_thresh = config.get("iou_thresh", 0.45) if config else 0.45
        self.imgsz = config.get("imgsz", 1024) if config else 1024
        self.agnostic_nms = config.get("agnostic_nms", True) if config else True

    def infer(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지 배열(Numpy)을 받아 레이아웃 영역의 Bounding Box 정보를 리스트로 반환합니다.
        """
        # YOLO 모델에 이미지 던지고 추론 실행 (고해상도, 임계값 조정)
        results = self.model.predict(
            source=image_np, 
            device=self.device, 
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            agnostic_nms=self.agnostic_nms,
            verbose=False
        )
        
        boxes_info = []
        # 결과에서 좌표(BBox), 신뢰도, 클래스 이름(제목, 사진 등) 긁어오기
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # 좌표 [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                boxes_info.append({
                    "label": cls_name,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                
        return boxes_info
