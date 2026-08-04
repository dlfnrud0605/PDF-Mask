import fitz
import re
from app.core.base import BaseModel

class PyMuPDFExtractor(BaseModel):
    """
    디지털 PDF에서 무거운 OCR 없이 초고속으로 텍스트와 Bounding Box를 추출하는 추출기.
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.target_width = 1280
        self._cache_page_num = -1
        self._cache_data = None

    def infer(self, *args, **kwargs):
        """BaseModel의 추상 메서드 구현 (직접 호출되지 않음)"""
        pass

    def extract_words(self, doc: fitz.Document, page_num: int):
        data = self._extract_rawdict(doc, page_num)
        return data["words"]

    def _extract_rawdict(self, doc: fitz.Document, page_num: int):
        if self._cache_page_num == page_num and self._cache_data is not None:
            return self._cache_data

        page = doc.load_page(page_num)
        page_dict = page.get_text("rawdict")
        
        pdf_to_1280_scale = self.target_width / max(page.rect.width, page.rect.height)
        
        BASELINE_MAX_RANGE = 8
        line_size_baselines = {}
        for bi, block in enumerate(page_dict.get('blocks', [])):
            if block.get('type') != 0: continue
            for li, line in enumerate(block.get('lines', [])):
                for span in line.get('spans', []):
                    size = round(span.get('size', 12), 1)
                    for char in span.get('chars', []):
                        c = char.get('c', '')
                        origin = char.get('origin')
                        if origin and c.strip() and c.isprintable():
                            line_size_baselines.setdefault((bi, li, size), []).append(origin[1])
        
        line_size_median = {}
        for key, vals in line_size_baselines.items():
            sv = sorted(vals)
            if sv[-1] - sv[0] <= BASELINE_MAX_RANGE:
                line_size_median[key] = sv[len(sv) // 2]

        all_words = []

        for bi, block in enumerate(page_dict.get('blocks', [])):
            if block.get('type') != 0: continue

            for li, line in enumerate(block.get('lines', [])):
                line_text = ""
                words_data = []
                word_chars = []
                word_text = ""
                
                for span in line.get('spans', []):
                    size = span.get('size', 12)
                    is_first_char_in_span = True
                    for char in span.get('chars', []):
                        c = char.get('c', '')
                        bbox = char.get('bbox')
                        
                        if c.strip() == "" or not c.isprintable():
                            if word_chars:
                                words_data.append({"word": word_text, "chars": word_chars})
                                word_chars, word_text = [], ""
                            line_text += " "
                            is_first_char_in_span = False
                        else:
                            origin = char.get('origin')
                            padding = size * 0.1
                            
                            if word_chars:
                                raw_gap = bbox[0] - word_chars[-1]['_raw_x2']
                                raw_min_size = min(word_chars[-1]['size'] / pdf_to_1280_scale, size)
                                
                                split = raw_gap > raw_min_size * 0.3
                                if is_first_char_in_span and not split:
                                    curr_baseline = origin[1] if origin else None
                                    baselines_raw = [ch['baseline_y'] / pdf_to_1280_scale for ch in word_chars if ch['baseline_y'] is not None]
                                    if baselines_raw and curr_baseline is not None:
                                        baselines_raw_sorted = sorted(baselines_raw)
                                        mid = len(baselines_raw_sorted) // 2
                                        median_baseline = baselines_raw_sorted[mid]
                                        baseline_diff = abs(curr_baseline - median_baseline)
                                        split = baseline_diff > 2
                                        
                                if split:
                                    words_data.append({"word": word_text, "chars": word_chars})
                                    word_chars, word_text = [], ""
                                    line_text += " "
                                    
                            shared_baseline = line_size_median.get((bi, li, round(size, 1)))
                            baseline_y = shared_baseline if shared_baseline is not None else (origin[1] if origin else None)
                            adj_bbox = [bbox[0] - padding, bbox[1], bbox[2] + padding, bbox[3]]
                            scaled_bbox = [v * pdf_to_1280_scale for v in adj_bbox]
                            scaled_baseline = baseline_y * pdf_to_1280_scale if baseline_y is not None else None
                            char_info = {"c": c, "bbox": scaled_bbox, "baseline_y": scaled_baseline, "size": size * pdf_to_1280_scale, "_raw_x2": bbox[2]}
                            word_chars.append(char_info)
                            word_text += c
                            line_text += c
                            is_first_char_in_span = False
                            
                if word_chars:
                    words_data.append({"word": word_text, "chars": word_chars})
                    
                if line_text.strip():
                    is_only_upper = bool(re.fullmatch(r'[A-Z\s]+', line_text))
                    
                    for w in words_data:
                        for ch in w['chars']:
                            if ch['baseline_y'] is not None:
                                s = ch['size']
                                by = ch['baseline_y']
                                top_margin = min(0.2 * s, 8)
                                ch['bbox'][1] = by - 0.8 * s - top_margin
                                if is_only_upper:
                                    bot_margin = min(0.15 * s, 6)
                                    ch['bbox'][3] = by + bot_margin
                                else:
                                    bot_margin = min(0.1 * s, 5)
                                    ch['bbox'][3] = by + 0.2 * s + bot_margin
                                    
                        wx = [c['bbox'][0] for c in w['chars']] + [c['bbox'][2] for c in w['chars']]
                        wy = [c['bbox'][1] for c in w['chars']] + [c['bbox'][3] for c in w['chars']]
                        
                        if wx and wy:
                            # 1px 단위 오차로 인한 부동소수점 처리
                            final_bbox = [int(min(wx)), int(min(wy)), int(max(wx)), int(max(wy))]
                            final_word_dict = {"text": w["word"], "bbox": final_bbox}
                            all_words.append(final_word_dict)

        self._cache_page_num = page_num
        self._cache_data = {"words": all_words}
        return self._cache_data
