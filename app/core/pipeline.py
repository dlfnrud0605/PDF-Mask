import os
import fitz
import random
from PIL import Image

from app.core.factory import ModelFactory
from app.utils.renderer import PDFRenderer

# analyze() 단계에서 OCR/스케일 계산용으로 이미 렌더링한 페이지 이미지를, 학습 데이터
# 보존용으로 한 번 더 렌더링하지 않고 그대로 저장해 두는 임시 폴더. 확정되면 여기서
# data/corrections/images/로 옮겨지고, 확정 안 되면 main.py에서 주기적으로 정리함.
ANALYZE_IMAGE_TMP_DIR = "data/tmp/analyze_images"

X_GAP_MAX_RATIO = 3.0  # 가로 간격이 평균 글자 높이의 이 배수를 넘으면 같은 행(row)이라도 다른 칸(컬럼)으로 보고 끊음

def is_same_line(box1, box2):
    """세로 위치만으로 같은 행(row)인지 판별 (가로 간격은 별도로 나중에 확인함)."""
    cy1 = (box1[1] + box1[3]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    h1 = box1[3] - box1[1]
    h2 = box2[3] - box2[1]
    avg_h = (h1 + h2) / 2

    if abs(cy1 - cy2) < avg_h * 0.3:
        return True
    overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    if overlap / max(h1, h2) > 0.3:
        return True
    return False

def unify_y_values_and_group(words):
    if not words: return []

    # 1단계: Y 중심값 기준으로 행(row) 클러스터링. 같은 행 여부는 세로 위치만
    # 보므로, 이 시점엔 아직 x축으로 서로 멀리 떨어진 여러 칸이 한 행에 섞여있을 수 있음.
    sorted_words = sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2)

    rows = []
    current_row = [sorted_words[0].copy()]
    for w in sorted_words[1:]:
        if is_same_line(current_row[-1]["bbox"], w["bbox"]):
            current_row.append(w.copy())
        else:
            rows.append(current_row)
            current_row = [w.copy()]
    if current_row:
        rows.append(current_row)

    # 2단계: 각 행 내부를 X좌표로 정렬해서 "진짜" 좌우 이웃 관계를 확정한 뒤,
    # 바로 옆 이웃끼리 가로 간격이 너무 크면(다른 칸일 가능성) 그 지점에서 끊어
    # 여러 칸(column) 그룹으로 분리함. (1단계에서 y중심으로 정렬했던 순서 그대로
    # 가로 간격을 비교하면, 서로 진짜 이웃이 아닌 임의의 두 항목을 비교하게 돼서
    # 같은 행의 먼 항목끼리도 잘못 끊기거나 순서가 뒤섞이는 문제가 있었음)
    normalized_words = []
    for row in rows:
        row.sort(key=lambda w: w["bbox"][0])

        col_groups = []
        current_col = [row[0]]
        for w in row[1:]:
            prev = current_col[-1]
            h = ((prev["bbox"][3] - prev["bbox"][1]) + (w["bbox"][3] - w["bbox"][1])) / 2
            x_gap = w["bbox"][0] - prev["bbox"][2]
            if x_gap > h * X_GAP_MAX_RATIO:
                col_groups.append(current_col)
                current_col = [w]
            else:
                current_col.append(w)
        col_groups.append(current_col)

        for col in col_groups:
            normalized_words.extend(col)

    return normalized_words

def dedupe_overlapping_words(words, iou_thresh=0.7):
    """
    같은 텍스트가 같은 위치에 겹쳐서 중복 추출된 경우(디지털 텍스트 모드에서 원본
    PDF/PPT가 굵게 보이려고 같은 텍스트를 같은 자리에 두 번 그려두는 경우 등)
    하나만 남기고 나머지는 버림. bbox가 실제로 많이 겹치고 텍스트까지 같아야만
    중복으로 판단함 (그냥 근처에 있는 서로 다른 어절은 안 건드림).
    """
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    kept = []
    for w in words:
        is_dup = False
        for k in kept:
            if k["text"] != w["text"]:
                continue
            ax1, ay1, ax2, ay2 = k["bbox"]
            bx1, by1, bx2, by2 = w["bbox"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = area(k["bbox"]) + area(w["bbox"]) - inter
                if union > 0 and inter / union > iou_thresh:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(w)
    return kept

def filter_contained_boxes(bboxes, iom_thresh=0.9):
    n = len(bboxes)
    remove = set()
    for i in range(n):
        if i in remove: continue
        ax1, ay1, ax2, ay2 = bboxes[i]
        area_a = (ax2 - ax1) * (ay2 - ay1)
        if area_a <= 0: continue
        
        for j in range(i + 1, n):
            if j in remove: continue
            bx1, by1, bx2, by2 = bboxes[j]
            area_b = (bx2 - bx1) * (by2 - by1)
            if area_b <= 0: continue
            
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                iom = inter_area / min(area_a, area_b)
                
                if iom >= iom_thresh:
                    if area_a < area_b:
                        remove.add(i)
                        break
                    else:
                        remove.add(j)
    return [i for i in range(n) if i not in remove]

def sort_by_reading_order(bboxes):
    n = len(bboxes)
    if n <= 1: return list(range(n))
    sorted_idx = sorted(range(n), key=lambda i: (bboxes[i][1] + bboxes[i][3]) / 2)
    lines = []
    current_line = [sorted_idx[0]]
    line_y = (bboxes[sorted_idx[0]][1] + bboxes[sorted_idx[0]][3]) / 2
    line_h = bboxes[sorted_idx[0]][3] - bboxes[sorted_idx[0]][1]
    for i in range(1, n):
        idx = sorted_idx[i]
        y_center = (bboxes[idx][1] + bboxes[idx][3]) / 2
        h = bboxes[idx][3] - bboxes[idx][1]
        threshold = min(line_h, h) * 0.5
        if abs(y_center - line_y) < threshold:
            current_line.append(idx)
        else:
            current_line.sort(key=lambda i: bboxes[i][0])
            lines.append(current_line)
            current_line = [idx]
            line_y = y_center
            line_h = h
    current_line.sort(key=lambda i: bboxes[i][0])
    lines.append(current_line)
    return [idx for line in lines for idx in line]


class PDFMaskingPipeline:
    def __init__(self, mode="ocr"):
        self.mode = mode
        self.renderer = PDFRenderer(target_width=1280)
        self.keyword_extractor = ModelFactory.get_processor("keyword_extractor")
        self.yolo_detector = ModelFactory.get_processor("layout_detector")

        if self.mode == "ocr":
            self.text_detector = ModelFactory.get_processor("text_detector")
            self.ocr_recognizer = ModelFactory.get_processor("recognizer")
        elif self.mode == "digital":
            self.pymupdf_extractor = ModelFactory.get_processor("pdf_extractor")
        else:
            raise ValueError("mode must be 'ocr' or 'digital'")
            
    def _extract_page_words(self, doc, input_pdf_path: str, page_num: int):
        """
        한 페이지의 어절 추출까지만 수행(청킹/NLP 병합 이전). 리전별 원본 어절 리스트와
        좌표 변환에 필요한 스케일/페이지 크기를 반환. process()와 analyze() 둘 다 이 원본
        어절에서 시작하고, 그 다음 필요한 쪽만 _chunk_words()로 청킹함.
        """
        page = doc[page_num]
        image_np = self.renderer.render_page(input_pdf_path, page_number=page_num)
        h, w = image_np.shape[:2]

        # 원본 PDF 픽셀 변환 스케일
        scale_x = page.rect.width / w
        scale_y = page.rect.height / h

        regions_words_data = []

        # 1. Layout Detection (두 모드 공통)
        regions = self.yolo_detector.infer(image_np)
        keep_idx = filter_contained_boxes([r['bbox'] for r in regions])
        regions = [regions[i] for i in keep_idx]
        order_idx = sort_by_reading_order([r['bbox'] for r in regions])
        regions = [regions[i] for i in order_idx]

        if self.mode == "ocr":
            # 2. Text Detection
            infer_out = self.text_detector.infer(image_np)
            full_word_results = infer_out[0] if isinstance(infer_out, tuple) else infer_out
            keep_idx = filter_contained_boxes([r['word_box'] for r in full_word_results])
            full_word_results = [full_word_results[i] for i in keep_idx]

            # 3. Text Recognition — 영역 배정 전, 탐지된 어절 전체에 대해 먼저 수행
            word_crops = []
            valid_word_boxes = []
            for res in full_word_results:
                gx1, gy1, gx2, gy2 = res['word_box']
                crop = image_np[max(0, gy1):min(h, gy2), max(0, gx1):min(w, gx2)]
                if crop.size == 0: continue
                word_crops.append(crop)
                valid_word_boxes.append([gx1, gy1, gx2, gy2])

            all_results = self.ocr_recognizer.infer(word_crops) if word_crops else []

            recognized_words = []
            seen_boxes = set()
            for idx, word_info in enumerate(all_results):
                if not word_info: continue
                text = "".join([item["text"] for item in word_info]).strip()
                if not text: continue
                bbox = valid_word_boxes[idx]
                box_key = tuple(bbox)
                if box_key in seen_boxes: continue
                seen_boxes.add(box_key)
                recognized_words.append({"text": text, "bbox": bbox})

            # 4. Assignment — 인식이 끝난 어절을 영역에 배정
            for region in regions:
                rx1, ry1, rx2, ry2 = region['bbox']
                region_words_with_text = []
                for w in recognized_words:
                    gx1, gy1, gx2, gy2 = w["bbox"]
                    cx = (gx1 + gx2) / 2
                    cy = (gy1 + gy2) / 2
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        region_words_with_text.append(w)

                regions_words_data.append({
                    "label": region['label'],
                    "region_bbox": [rx1, ry1, rx2, ry2],
                    "words": unify_y_values_and_group(region_words_with_text)
                })

        elif self.mode == "digital":
            # 2. Text Extraction — PyMuPDF는 탐지와 인식이 한 번에 이뤄짐(이미 텍스트를 알고 있음)
            extracted_words = self.pymupdf_extractor.extract_words(doc, page_num)
            extracted_words = dedupe_overlapping_words(extracted_words)

            # 3. Assignment — 추출된 어절을 YOLO 영역에 배정
            for region in regions:
                rx1, ry1, rx2, ry2 = region['bbox']
                region_words_with_text = []

                for w in extracted_words:
                    gx1, gy1, gx2, gy2 = w['bbox']
                    cx = (gx1 + gx2) / 2
                    cy = (gy1 + gy2) / 2

                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        region_words_with_text.append(w)

                regions_words_data.append({
                    "label": region['label'],
                    "region_bbox": [rx1, ry1, rx2, ry2],
                    "words": unify_y_values_and_group(region_words_with_text)
                })

        return regions_words_data, scale_x, scale_y, page.rect.width, page.rect.height, image_np

    def _chunk_words(self, regions_words_data):
        """리전별 원본 어절을 NLP 청킹(형태소 분석 및 띄어쓰기 교정 적용)해서 청크로 병합."""
        if hasattr(self.keyword_extractor, "infer_pdf"):
            return self.keyword_extractor.infer_pdf([regions_words_data])[0]
        return self.keyword_extractor.infer(regions_words_data)

    def analyze(self, input_pdf_path: str, file_id: str | None = None) -> list[dict]:
        """
        마스킹을 확정하지 않고, 페이지별 원본 어절 위치(bbox, PDF pt 단위)+점수를 반환.
        어절은 항상 원본 그대로 개별로 주되(자동 청킹이 서로 다른 칸의 텍스트를 잘못
        합치는 문제가 있어 최종 판단은 프론트/사용자에게 맡김), 자동 청킹 결과는
        "이 어절들을 묶으면 이렇게 된다"는 제안(groups)으로 같이 내려줌 — 프론트에서
        사용자가 직접 Shift+클릭으로 묶은 것과 동일한 형식(어절 id 리스트)이라,
        기본값으로 미리 묶인 채로 보여주고 사용자가 틀린 부분만 고칠 수 있게 함.

        file_id가 있으면, 이미 OCR/스케일 계산용으로 렌더링해 둔 페이지 이미지를 학습
        데이터 보존용으로 그대로 저장함(추가 렌더링 없음) — 나중에 사용자가 청킹을
        확정하면 이 이미지를 그대로 정답 데이터 옆으로 옮겨서 씀.
        """
        random.seed(51)
        doc = fitz.open(input_pdf_path)
        pages_out = []
        try:
            total_pages = len(doc)
            for page_num in range(total_pages):
                regions_words_data, scale_x, scale_y, page_w, page_h, image_np = self._extract_page_words(
                    doc, input_pdf_path, page_num
                )
                if file_id:
                    img_dir = os.path.join(ANALYZE_IMAGE_TMP_DIR, file_id)
                    os.makedirs(img_dir, exist_ok=True)
                    Image.fromarray(image_np).save(os.path.join(img_dir, f"p{page_num}.jpg"), "JPEG", quality=85)

                keyword_results = self._chunk_words(regions_words_data)

                chunks_out = []
                groups_out = []
                chunk_idx = 0
                for region_idx, region in enumerate(regions_words_data):
                    # (chunk_id, 원본 픽셀 bbox) — 이 리전 어절들이 어느 청크에 속하는지
                    # 기하학적으로(청크의 합집합 bbox 안에 중심점이 들어가는지) 매칭하는 데 씀
                    region_word_ids = []
                    for w in region.get("words", []):
                        x1, y1, x2, y2 = w["bbox"]
                        chunk_id = f"p{page_num}_c{chunk_idx}"
                        chunks_out.append({
                            "chunk_id": chunk_id,
                            "text": w["text"],
                            "bbox": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                            "score": random.random(),
                        })
                        region_word_ids.append((chunk_id, w["bbox"]))
                        chunk_idx += 1

                    region_chunks = (
                        keyword_results.get(region_idx, [])
                        if isinstance(keyword_results, dict)
                        else (keyword_results[region_idx] if region_idx < len(keyword_results) else [])
                    )
                    claimed = set()
                    for kw in region_chunks:
                        kx1, ky1, kx2, ky2 = kw["bbox"]
                        group_ids = []
                        for wid, bbox in region_word_ids:
                            if wid in claimed:
                                continue
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            if kx1 <= cx <= kx2 and ky1 <= cy <= ky2:
                                group_ids.append(wid)
                                claimed.add(wid)
                        if len(group_ids) >= 2:
                            groups_out.append(group_ids)

                pages_out.append({
                    "page_num": page_num,
                    "width": page_w,
                    "height": page_h,
                    "chunks": chunks_out,
                    "groups": groups_out,
                })
        finally:
            doc.close()

        return pages_out
