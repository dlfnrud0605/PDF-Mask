import cv2
import numpy as np

def save_prob_map(prob_map: np.ndarray, save_path: str):
    """
    DBNet++의 Probability Map을 Heatmap으로 변환하여 저장함.
    """
    try:
        if prob_map is None:
            return
        map_img = (prob_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap)
    except Exception as e:
        print(f"[Visualizer] 맵 저장 중 오류 발생: {e}")

def save_detection_result(img_orig: np.ndarray, save_path: str, boxes: list = None):
    """
    검출된 박스들(Word Boxes)을 원본 이미지에 그려서 저장함.
    """
    try:
        if img_orig is None:
            return
            
        H, W = img_orig.shape[:2]
        
        # 원본 이미지 복사 및 BGR 변환
        vis_img = img_orig.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        elif vis_img.shape[2] == 4:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGRA2BGR)
            
        # 박스 그리기
        if boxes:
            for i, box in enumerate(boxes):
                # box 형식이 [x1, y1, x2, y2]인 경우 처리
                if len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    # 녹색 박스로 시각화
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        # 파일 저장
        cv2.imwrite(save_path, vis_img)
        
    except Exception as e:
        print(f"[Visualizer] 검출 시각화 저장 중 오류 발생: {e}")
