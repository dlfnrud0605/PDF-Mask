import os
import fitz  # PyMuPDF
import json
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import shutil
from collections import Counter

# --- 설정 (dataset/ 폴더 내부 기준으로 변경) ---
# 스크립트 위치를 기준으로 프로젝트 루트를 찾음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_ROOT = BASE_DIR  # 현재 폴더(dataset/)가 데이터셋 루트
RAW_PDF_DIR = os.path.join(DATASET_ROOT, "raw")
PROCESSED_PDF_DIR = os.path.join(DATASET_ROOT, "processed")
IMG_OUT_DIR = os.path.join(DATASET_ROOT, "images")
ANN_OUT_DIR = os.path.join(DATASET_ROOT, "annotations") # 페이지별 임시 JSON 저장
ANN_VERT_DIR = os.path.join(DATASET_ROOT, "annotations_vertical") # 세로 텍스트 포함 페이지
IMG_VERT_DIR = os.path.join(DATASET_ROOT, "images_vertical")       # 세로 텍스트 포함 이미지

DPI_RENDER = 300                  # 고화질 렌더링용 (메모리상)
TARGET_WIDTH = 1280               # 최종 저장 이미지 너비
PREVIEW_WIDTH = 1200              # GUI 미리보기 너비
# --- 설정 끝 ---

class IntegratedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("통합 OCR 데이터셋 빌더 v2.0 (dataset/ 전용)")
        
        # 폴더 생성
        for d in [IMG_OUT_DIR, ANN_OUT_DIR, PROCESSED_PDF_DIR, IMG_VERT_DIR, ANN_VERT_DIR]:
            os.makedirs(d, exist_ok=True)
            
        if not os.path.exists(RAW_PDF_DIR):
            messagebox.showerror("오류", f"'{RAW_PDF_DIR}' 폴더가 없습니다. 프로젝트 루트에 'raw' 폴더를 만들어주세요.")
            self.root.destroy()
            return

        self.pdf_list = ([f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")])
        if not self.pdf_list:
            messagebox.showinfo("알림", f"'{RAW_PDF_DIR}' 폴더에 PDF 파일이 없습니다.")
            self.root.destroy()
            return
            
        self.current_pdf_idx = 0
        self.current_page_idx = 0
        self.doc = None
        self.show_dbnet_sim = False  # 기본적으로 원본 박스만 표시, k로 토글
        self.show_char_boxes = False  # 글자 개별 bbox 표시, c로 토글
        self.history = []  # (pdf_idx, page_idx) 스택 — 이전 페이지로 돌아가기용
        self.selected_ks = set()  # 다중 선택된 word 인덱스(k) 집합
        self._rubber_band_id = None
        self._rubber_band_start = None
        
        try:
            self.root.attributes('-zoomed', True) # Linux/Windows 최대화
        except Exception:
            try:
                self.root.state('zoomed')
            except:
                pass
                
        self.setup_ui()
        self.root.update() # 창 크기 계산을 위해 강제 렌더링
        self.load_next_page()

    def _get_korean_font(self, size=10, bold=False):
        """
        한글+라틴 모두 지원하는 TrueType 폰트를 반환.
        Linux Tk는 X11 코어 폰트만 인식하므로:
          1단계: ~/.fonts_x11에 NanumGothic.ttf X11 인덱스를 생성하고 경로 등록
          2단계: X11 비트맵 한글 폰트(gulim 등) 폴백 — 단, 라틴 문자가 Symbol 폰트로
                 잘못 대체될 수 있으므로 TrueType이 우선임
        """
        import subprocess as _sp
        import shutil as _sh
        weight = 'bold' if bold else 'normal'

        fonts_dir = os.path.join(os.path.expanduser('~'), '.fonts_x11')
        src_nanum = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        src_nanum_bold = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'

        # NanumGothic TrueType X11 인덱스가 없으면 생성
        if not os.path.exists(os.path.join(fonts_dir, 'fonts.dir')):
            try:
                os.makedirs(fonts_dir, exist_ok=True)
                for src in [src_nanum, src_nanum_bold]:
                    if os.path.exists(src):
                        dst = os.path.join(fonts_dir, os.path.basename(src))
                        if not os.path.exists(dst):
                            _sh.copy2(src, dst)
                _sp.run(['mkfontscale', fonts_dir], capture_output=True, timeout=5)
                _sp.run(['mkfontdir', fonts_dir], capture_output=True, timeout=5)
            except Exception:
                pass

        # X11 폰트 경로 등록
        try:
            _sp.run(['xset', 'fp+', fonts_dir], capture_output=True, timeout=3)
            _sp.run(['xset', 'fp+', '/usr/share/fonts/X11/75dpi'],
                    capture_output=True, timeout=3)
            _sp.run(['xset', 'fp', 'rehash'], capture_output=True, timeout=3)
        except Exception:
            pass

        available = set(tkfont.families())
        # TrueType NanumGothic 우선 (한글+라틴 모두 올바르게 렌더링)
        # X11 비트맵 한글 폰트는 라틴 문자를 Symbol 폰트로 잘못 치환하는 문제 있음
        preferred = ['nanumgothic', 'NanumGothic', 'gulim', 'dotum', 'batang']
        for name in preferred:
            if name in available:
                return (name, size, weight)

        return ('TkDefaultFont', size, weight)

    def setup_ui(self):
        # ttk 전체 위젯 한글 폰트 적용 (Linux 기본폰트가 한글 미지원인 경우 대비)
        btn_font  = self._get_korean_font(size=10)
        lbl_font  = self._get_korean_font(size=12, bold=True)
        list_font = self._get_korean_font(size=10)
        style = ttk.Style()
        style.configure('TButton', font=btn_font)
        style.configure('TLabel',  font=btn_font)

        # 상단 정보바
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.lbl_status = ttk.Label(self.info_frame, text="준비 중...", font=lbl_font)
        self.lbl_status.pack(side=tk.LEFT)
        
        # 메인 레이아웃 (PanedWindow)
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 1. 왼쪽: 이미지 캔버스
        self.canvas_frame = ttk.Frame(self.paned)
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(self.canvas_frame, bg="white",
                               xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.paned.add(self.canvas_frame, stretch="always")
        
        # 2. 오른쪽: 텍스트 리스트
        self.list_frame = ttk.Frame(self.paned)
        self.list_scroll = ttk.Scrollbar(self.list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(self.list_frame, yscrollcommand=self.list_scroll.set,
                                 width=25, font=list_font)
        self.list_scroll.config(command=self.listbox.yview)
        self.list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.paned.add(self.list_frame, stretch="never", width=200)
        
        # 하단 버튼바
        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.btn_prev = ttk.Button(self.btn_frame, text="← 이전 (A)", command=self.go_back)
        self.btn_prev.pack(side=tk.LEFT, padx=10, ipady=10)

        self.btn_save = ttk.Button(self.btn_frame, text="저장 및 다음 (S)", command=self.save_and_next)
        self.btn_save.pack(side=tk.LEFT, padx=10, ipady=10)

        self.btn_skip = ttk.Button(self.btn_frame, text="다음 (D)", command=self.skip_and_next)
        self.btn_skip.pack(side=tk.LEFT, padx=10, ipady=10)

        self.btn_vert = ttk.Button(self.btn_frame, text="세로 저장 (W)", command=self.save_vertical_and_next)
        self.btn_vert.pack(side=tk.LEFT, padx=10, ipady=10)

        self.btn_delete = ttk.Button(self.btn_frame, text="삭제 (Q)", command=self.delete_current_page)
        self.btn_delete.pack(side=tk.LEFT, padx=10, ipady=10)

        self.btn_export = ttk.Button(self.btn_frame, text="MMOCR 최종 병합", command=self.export_to_mmocr)
        self.btn_export.pack(side=tk.RIGHT, padx=10)

        # 단축키 바인딩
        self.root.bind('a', lambda e: self.go_back())
        self.root.bind('s', lambda e: self.save_and_next())
        self.root.bind('d', lambda e: self.skip_and_next())
        self.root.bind('w', lambda e: self.save_vertical_and_next())
        self.root.bind('q', lambda e: self.delete_current_page())
        self.root.bind('k', self.toggle_dbnet_sim)
        self.root.bind('c', self.toggle_char_boxes)
        self.root.bind('<Control-g>', self.group_selected)
        self.root.bind('<Control-G>', self.group_selected)
        self.root.bind('<Control-Shift-g>', self.ungroup_selected)
        self.root.bind('<Control-Shift-G>', self.ungroup_selected)
        self.listbox.bind("<Delete>", self.on_delete_word)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_word)

        # 캔버스 pan / zoom 바인딩
        self.canvas.bind("<ButtonPress-1>",   self._on_canvas_press)
        self.canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom_scroll)   # Windows
        self.canvas.bind("<Control-Button-4>", lambda e: self._apply_zoom(1.1))  # Linux
        self.canvas.bind("<Control-Button-5>", lambda e: self._apply_zoom(0.9))

    def load_next_page(self):
        if self.doc is None:
            if self.current_pdf_idx >= len(self.pdf_list):
                messagebox.showinfo("완료", "모든 PDF 처리가 완료되었습니다.")
                self.root.destroy()
                return
            
            pdf_name = self.pdf_list[self.current_pdf_idx]
            raw_path  = os.path.join(RAW_PDF_DIR,       pdf_name)
            proc_path = os.path.join(PROCESSED_PDF_DIR, pdf_name)
            if os.path.exists(raw_path):
                self.doc = fitz.open(raw_path)
            elif os.path.exists(proc_path):
                self.doc = fitz.open(proc_path)
            else:
                messagebox.showerror("오류", f"PDF 파일을 찾을 수 없습니다:\n{pdf_name}")
                self.current_pdf_idx += 1
                self.load_next_page()
                return
            self.current_page_idx = 0
            
        if self.current_page_idx >= len(self.doc):
            pdf_name = self.pdf_list[self.current_pdf_idx]
            try:
                self.doc.close()
                # RAW_PDF_DIR에 있을 때만 이동 (go_back으로 열린 경우 이미 이동됐을 수 있음)
                raw_path = os.path.join(RAW_PDF_DIR, pdf_name)
                if os.path.exists(raw_path):
                    shutil.move(raw_path, os.path.join(PROCESSED_PDF_DIR, pdf_name))
            finally:
                self.doc = None
            self.current_pdf_idx += 1
            self.load_next_page()
            return

        self.process_page()

    def process_page(self):
        self.selected_ks.clear()
        pdf_name = self.pdf_list[self.current_pdf_idx]
        self.lbl_status.config(text=f"파일: {pdf_name} | 페이지: {self.current_page_idx + 1} / {len(self.doc)}")
        
        page = self.doc.load_page(self.current_page_idx)
        
        # 1. 고화질 렌더링 및 1280px 리사이즈 (메모리)
        pix = page.get_pixmap(dpi=DPI_RENDER)
        img_300 = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 긴쪽 1280px로 조정
        scale_to_1280 = TARGET_WIDTH / max(img_300.width, img_300.height)
        new_w = int(img_300.width * scale_to_1280)
        new_h = int(img_300.height * scale_to_1280)
        self.master_img = img_300.resize((new_w, new_h), Image.LANCZOS)
        self.box_color = self._complement_color(self.master_img)
        self._zoom_level = 1.0

        # 2. 데이터 추출 (72dpi -> 1280px 좌표계)
        pdf_w = page.rect.width
        self.pdf_to_1280_scale = TARGET_WIDTH / max(page.rect.width, page.rect.height)
        
        self.raw_data = self.extract_text_data(page)

        # DEBUG: PyMuPDF 원본 rawdict 출력
        print("\n" + "="*60)
        print(f"[RAW] {self.pdf_list[self.current_pdf_idx]} p{self.current_page_idx}")
        print("="*60)
        page_dict = page.get_text("rawdict")
        for bi, block in enumerate(page_dict.get('blocks', [])):
            if block.get('type') != 0: continue
            for li, line in enumerate(block.get('lines', [])):
                for si, span in enumerate(line.get('spans', [])):
                    chars_info = [(ch.get('c',''), ch.get('bbox'), ch.get('origin')) for ch in span.get('chars', [])]
                    print(f"  block[{bi}] line[{li}] span[{si}] size={span.get('size',0):.1f}: {chars_info}")
        print("="*60 + "\n")

        # 완전 동일 좌표 중복 단어 자동 제거
        self._remove_exact_duplicates()

        # DEBUG: clipping 전 원본 line_bbox 저장
        self._orig_line_bboxes = [line['line_bbox'][:] for line in self.raw_data]
        # 겹침 깎기 로직 비활성화 (모두 descender 적용)
        # self.apply_vertical_clipping()
        # self.apply_mid_point_clipping()
        
        # 3. 미리보기 준비
        preview_scale = PREVIEW_WIDTH / new_w
        p_h = int(new_h * preview_scale)
        preview_img = self.master_img.resize((PREVIEW_WIDTH, p_h), Image.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(preview_img)
        self.canvas_scale = preview_scale
        
        self.refresh_ui_data()
        if self.red_set and self.yellow_set:
            self.root.after(150, lambda: (self._notify_overlap(), self._notify_jamo()))
        elif self.red_set:
            self.root.after(150, self._notify_overlap)
        elif self.yellow_set:
            self.root.after(150, self._notify_jamo)

    def _notify_overlap(self):
        count = len(self.red_set)
        messagebox.showwarning(
            "겹침 감지",
            f"이 페이지에 겹치는 단어 박스가 {count}개 있습니다.\n"
            "빨간색으로 표시된 항목을 확인하고 필요 시 삭제(Delete)해주세요."
        )

    def _notify_jamo(self):
        words = [self.raw_data[li]['words'][wi]['word']
                 for k, (li, wi) in enumerate(self.word_map)
                 if k in self.yellow_set]
        preview = ", ".join(f"'{w}'" for w in words[:10])
        messagebox.showinfo(
            "단일 자모 감지",
            f"이 페이지에 단독 자모 박스가 {len(self.yellow_set)}개 있습니다: {preview}\n"
            "노란색으로 표시됩니다. 불필요한 경우 삭제(Delete)해주세요."
        )

    def extract_text_data(self, page):
        # PyMuPDF rawdict 추출
        page_dict = page.get_text("rawdict")

        # 1차 패스: (block, line) × font_size 별 공유 baseline(중앙값) 사전 계산
        # 같은 줄 안에서 font size가 같은 글자들은 동일한 baseline을 써서 bbox 상하를 맞춤
        # baseline 범위가 너무 크면(곡선 텍스트 등) 적용 안 함
        BASELINE_MAX_RANGE = 8  # 이 px 이내일 때만 공유 baseline 적용 (72dpi 기준)
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
            if sv[-1] - sv[0] <= BASELINE_MAX_RANGE:  # 범위 작을 때만 적용
                line_size_median[key] = sv[len(sv) // 2]

        data = []
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
                        bbox = char.get('bbox')  # 72dpi

                        if c.strip() == "" or not c.isprintable():  # 공백·제어문자 단어 마감
                            if word_chars:
                                words_data.append({"word": word_text, "chars": word_chars})
                                word_chars, word_text = [], ""
                            line_text += " "
                            is_first_char_in_span = False
                        else:
                            origin = char.get('origin')
                            padding = size * 0.1

                            # 단어 분리: 1차 baseline 비교(span 경계에서만), 2차 gap 비교
                            if word_chars:
                                raw_gap = bbox[0] - word_chars[-1]['_raw_x2']
                                raw_min_size = min(word_chars[-1]['size'] / self.pdf_to_1280_scale, size)

                                split = raw_gap > raw_min_size * 0.3
                                if is_first_char_in_span and not split:
                                    curr_baseline = origin[1] if origin else None
                                    baselines_raw = [ch['baseline_y'] / self.pdf_to_1280_scale for ch in word_chars if ch['baseline_y'] is not None]
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

                            # 같은 줄 + 같은 font size → 공유 median baseline 사용 (bbox 상하 통일)
                            shared_baseline = line_size_median.get((bi, li, round(size, 1)))
                            baseline_y = shared_baseline if shared_baseline is not None else (origin[1] if origin else None)
                            adj_bbox = [bbox[0] - padding, bbox[1], bbox[2] + padding, bbox[3]]
                            scaled_bbox = [v * self.pdf_to_1280_scale for v in adj_bbox]
                            scaled_baseline = baseline_y * self.pdf_to_1280_scale if baseline_y is not None else None
                            char_info = {"c": c, "bbox": scaled_bbox, "baseline_y": scaled_baseline, "size": size * self.pdf_to_1280_scale, "_raw_x2": bbox[2]}
                            word_chars.append(char_info)
                            word_text += c
                            line_text += c
                            is_first_char_in_span = False
                
                if word_chars:
                    words_data.append({"word": word_text, "chars": word_chars})

                if line_text.strip():
                    import re
                    # 오직 영어 대문자(A-Z)와 공백(스페이스 등)으로만 이루어져 있을 때만 True
                    is_only_upper = bool(re.fullmatch(r'[A-Z\s]+', line_text))
                    
                    for w in words_data:
                        # 모든 글자에 먼저 베이스라인 기반 박스 적용
                        for ch in w['chars']:
                            if ch['baseline_y'] is not None:
                                s = ch['size']
                                by = ch['baseline_y']
                                
                                # 상단: 기본 글꼴 높이(0.8) + 여백(최대 8px 제한)
                                top_margin = min(0.2 * s, 8)
                                ch['bbox'][1] = by - 0.8 * s - top_margin
                                
                                # 하단: 순수 영어 대문자일 때만 타이트하게, 한글이나 소문자가 있으면 넉넉한 여백 추가
                                if is_only_upper:
                                    bot_margin = min(0.15 * s, 6)
                                    ch['bbox'][3] = by + bot_margin
                                else:
                                    bot_margin = min(0.1 * s, 5)
                                    ch['bbox'][3] = by + 0.2 * s + bot_margin
                        
                    # 라인 전체 bbox 계산
                    all_x = [c['bbox'][0] for w in words_data for c in w['chars']] + \
                            [c['bbox'][2] for w in words_data for c in w['chars']]
                    all_y = [c['bbox'][1] for w in words_data for c in w['chars']] + \
                            [c['bbox'][3] for w in words_data for c in w['chars']]
                    l_bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]
                    
                    data.append({
                        "text": line_text.strip(),
                        "words": words_data,
                        "line_bbox": l_bbox
                    })
        return data

    def _remove_exact_duplicates(self):
        """텍스트와 bbox가 완전히 동일한(1px 오차 허용) 중복 단어를 제거."""
        seen = {}  # key: (text, x0_r, y0_r, x1_r, y1_r) → True
        for line in self.raw_data:
            survivors = []
            for word in line['words']:
                wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                if not wx: continue
                # 1px 단위로 반올림해서 부동소수점 오차 흡수
                key = (word['word'], round(min(wx)), round(min(wy)), round(max(wx)), round(max(wy)))
                if key in seen:
                    continue
                seen[key] = True
                survivors.append(word)
            line['words'] = survivors
        # 단어가 없어진 라인 제거
        self.raw_data = [line for line in self.raw_data if line['words']]

    def apply_vertical_clipping(self):
        """단어 단위 수직 겹침 제거 (줄 단위 병합 오류 방지)"""
        all_words = []
        for line in self.raw_data:
            for word in line['words']:
                wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                if not wx or not wy: continue
                word['temp_bbox'] = [min(wx), min(wy), max(wx), max(wy)]
                all_words.append(word)

        n = len(all_words)
        if n < 2: return

        all_words.sort(key=lambda w: w['temp_bbox'][1])
        top_trims = [0.0] * n

        for i in range(n):
            wi = all_words[i]
            
            # 1. 윗단어(wi)와 수평으로 겹치는 모든 아래 단어들 찾기
            overlapping_below = []
            for j in range(n):
                if j == i: continue
                wj = all_words[j]
                if wj['temp_bbox'][1] <= wi['temp_bbox'][1]: continue
                
                # 수평 겹침 확인
                x_ol = min(wi['temp_bbox'][2], wj['temp_bbox'][2]) - max(wi['temp_bbox'][0], wj['temp_bbox'][0])
                if x_ol > 0:
                    overlapping_below.append((j, wj))
            
            if not overlapping_below: continue
            
            # 2. 그 중 가장 가까운 y좌표(바로 아랫줄) 찾기
            best_y0 = min([wj['temp_bbox'][1] for j, wj in overlapping_below])
            
            # 3. 바로 아랫줄(best_y0와 유사한 y좌표)에 속하면서 수평으로 겹치는 '모든' 단어 깎기
            for j, wj in overlapping_below:
                if abs(wj['temp_bbox'][1] - best_y0) <= 5: # 5px 이내면 같은 줄로 간주
                    v_overlap = wi['temp_bbox'][3] - wj['temp_bbox'][1]
                    if v_overlap > 0:
                        top_trims[j] = max(top_trims[j], v_overlap)

        # 깎인 높이 적용
        for i, word in enumerate(all_words):
            t = top_trims[i]
            if t == 0: continue
            
            new_y0 = word['temp_bbox'][1] + t
            if new_y0 >= word['temp_bbox'][3]: continue

            for ch in word['chars']:
                ch['bbox'][1] = max(ch['bbox'][1], new_y0)

    def apply_mid_point_clipping(self):
        """글자 간 겹침 방지 (Mid-X Clipping)"""
        for line in self.raw_data:
            all_chars = []
            for word in line['words']:
                for char in word['chars']:
                    all_chars.append(char)
            
            all_chars.sort(key=lambda c: c['bbox'][0])
            for i in range(len(all_chars) - 1):
                c1, c2 = all_chars[i], all_chars[i+1]
                if c1['bbox'][2] > c2['bbox'][0]: # 겹침
                    mid = (c1['bbox'][2] + c2['bbox'][0]) / 2
                    c1['bbox'][2] = mid
                    c2['bbox'][0] = mid

    def refresh_ui_data(self):
        self.listbox.delete(0, tk.END)
        self.word_map = []
        
        # 중복 겹침 단어 집합(red_set) 찾기
        all_words_info = []
        idx = 0
        for li, line in enumerate(self.raw_data):
            for wi, word in enumerate(line['words']):
                wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                if not wx or not wy: continue
                wb = [min(wx), min(wy), max(wx), max(wy)]
                all_words_info.append({'k': idx, 'text': word['word'], 'wb': wb})
                idx += 1
                
        def _overlap(b1, b2):
            ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            inter = ix * iy
            if inter == 0: return 0.0
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0
            containment = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
            return max(iou, containment)

        self.red_set = set()
        for i in range(len(all_words_info)):
            for j in range(i + 1, len(all_words_info)):
                w1 = all_words_info[i]
                w2 = all_words_info[j]
                if _overlap(w1['wb'], w2['wb']) >= 0.4:
                    self.red_set.add(w1['k'])
                    self.red_set.add(w2['k'])

        # 한글 자모(ㄱ~ㅎ, ㅏ~ㅣ)가 포함된 단어 탐지 (단독 자모 or 자모 분해된 단어)
        JAMO = set("ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
        self.yellow_set = set()
        for wi in all_words_info:
            text = wi['text'].replace(' ', '')
            if text and any(c in JAMO for c in text):
                self.yellow_set.add(wi['k'])

        k = 0
        for li, line in enumerate(self.raw_data):
            for wi, word in enumerate(line['words']):
                self.listbox.insert(tk.END, f"[{k+1:03d}] {word['word']}")
                if k in self.red_set:
                    self.listbox.itemconfig(k, {'fg': 'red'})
                elif k in self.yellow_set:
                    self.listbox.itemconfig(k, {'fg': '#CC8800'})
                self.word_map.append((li, wi))
                k += 1
        self._redraw_canvas()

    def _redraw_canvas(self):
        # 캔버스의 실제 크기 가져오기
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # UI가 아직 렌더링되지 않았을 경우를 대비한 안전 장치
        if cw < 10 or ch < 10:
            cw = self.root.winfo_width() - 220
            ch = self.root.winfo_height() - 100
        if cw < 10 or ch < 10:
            cw, ch = PREVIEW_WIDTH, 800

        # 긴 쪽 기준으로 화면에 꽉 차게(Fit) 스케일 계산
        scale_w = cw / self.master_img.width
        scale_h = ch / self.master_img.height
        fit_scale = min(scale_w, scale_h) * 0.98 # 2% 여백
        
        new_w = int(self.master_img.width * fit_scale * self._zoom_level)
        new_h = int(self.master_img.height * fit_scale * self._zoom_level)
        
        preview = self.master_img.resize((new_w, new_h), Image.BILINEAR)
        self.preview_tk = ImageTk.PhotoImage(preview)
        self.canvas_scale = new_w / self.master_img.width

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_tk)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

        # DEBUG: clipping 전 원본 line_bbox (회색 점선)
        for ob in getattr(self, '_orig_line_bboxes', []):
            s = self.canvas_scale
            self.canvas.create_rectangle(ob[0]*s, ob[1]*s, ob[2]*s, ob[3]*s,
                                         outline="gray", width=1, dash=(4, 4), tags="debug_orig")

        k = 0
        for li, line in enumerate(self.raw_data):
            for wi, word in enumerate(line['words']):
                wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                if not wx or not wy: continue
                wb = [min(wx) * self.canvas_scale, min(wy) * self.canvas_scale,
                      max(wx) * self.canvas_scale, max(wy) * self.canvas_scale]
                
                if k in getattr(self, 'red_set', set()):
                    outline_color = "red"
                elif k in getattr(self, 'yellow_set', set()):
                    outline_color = "#FFAA00"
                else:
                    outline_color = self.box_color
                self.canvas.create_rectangle(wb[0], wb[1], wb[2], wb[3],
                                             outline=outline_color, width=1, tags=f"word_{k}")
                if k in getattr(self, 'selected_ks', set()):
                    self.canvas.create_rectangle(wb[0]-2, wb[1]-2, wb[2]+2, wb[3]+2,
                                                 outline="cyan", width=2, tags=f"sel_{k}")
                
                if self.show_dbnet_sim:
                    # DBNet Threshold Map (임계값 지도) 시뮬레이션 (r=0.65)
                    # 원본 박스를 D만큼 밖으로 팽창시킵니다.
                    w_orig, h_orig = max(wx) - min(wx), max(wy) - min(wy)
                    r_thresh = 0.65
                    d_orig_thresh = (w_orig * h_orig * (1 - r_thresh**2)) / (2 * (w_orig + h_orig)) if (w_orig + h_orig) > 0 else 0
                    d_thresh = d_orig_thresh * self.canvas_scale
                    
                    # 팽창된 박스 (Threshold Map의 바깥쪽 경계선) - 하늘색 점선
                    self.canvas.create_rectangle(wb[0] - d_thresh, wb[1] - d_thresh, wb[2] + d_thresh, wb[3] + d_thresh,
                                                 outline="#00FFFF", width=1, dash=(2, 2), tags=f"word_{k}_thresh")

                    # DBNet 공식 기반 다중 수축 박스 시뮬레이션 (r=0.85, 0.75, 0.65)
                    for r_val, color in zip([0.85, 0.75, 0.65], ["#00FF00", "#FFA500", "#FF00FF"]):
                        d_orig = (w_orig * h_orig * (1 - r_val**2)) / (2 * (w_orig + h_orig)) if (w_orig + h_orig) > 0 else 0
                        d = d_orig * self.canvas_scale
                        self.canvas.create_rectangle(wb[0] + d, wb[1] + d, wb[2] - d, wb[3] - d,
                                                     outline=color, width=1, tags=f"word_{k}_r{r_val}")

                # 글자 개별 bbox 그리기 (c키 토글)
                if self.show_char_boxes:
                    for ci, ch in enumerate(word['chars']):
                        cb = ch['bbox']
                        self.canvas.create_rectangle(
                            cb[0] * self.canvas_scale, cb[1] * self.canvas_scale,
                            cb[2] * self.canvas_scale, cb[3] * self.canvas_scale,
                            outline="#FF6600", width=1, tags=f"char_{k}_{ci}"
                        )

                # 베이스라인 그리기 (빨간 점선)
                baselines = [c.get('baseline_y') for c in word['chars'] if c.get('baseline_y') is not None]
                if baselines:
                    bly = baselines[0] * self.canvas_scale
                    self.canvas.create_line(wb[0], bly, wb[2], bly, fill="red", dash=(2, 2), tags=f"baseline_{k}")
                
                k += 1

        # 줌(Zoom) 등으로 캔버스가 다시 그려질 때 기존 선택(Highlight) 유지
        self.on_select_word(None)

    def toggle_dbnet_sim(self, event=None):
        self.show_dbnet_sim = not self.show_dbnet_sim
        self._redraw_canvas()

    def toggle_char_boxes(self, event=None):
        self.show_char_boxes = not self.show_char_boxes
        self._redraw_canvas()

    # -------------------------------------------------- pan / zoom 이벤트 --
    def _on_canvas_press(self, event):
        self._drag_start  = (event.x, event.y)
        self._drag_moved  = False
        self._zoom_last_y = event.y
        self._rubber_band_start = None
        shift = event.state & 0x1
        ctrl  = event.state & 0x4
        if shift:
            self._rubber_band_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        elif not ctrl:
            self.canvas.scan_mark(event.x, event.y)

    def _on_canvas_drag(self, event):
        if abs(event.x - self._drag_start[0]) > 3 or abs(event.y - self._drag_start[1]) > 3:
            self._drag_moved = True
        shift = event.state & 0x1
        ctrl  = event.state & 0x4
        if shift and self._rubber_band_start:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            if self._rubber_band_id:
                self.canvas.delete(self._rubber_band_id)
            x0, y0 = self._rubber_band_start
            self._rubber_band_id = self.canvas.create_rectangle(
                x0, y0, cx, cy, outline="cyan", width=2, dash=(4, 4), tags="rubber_band"
            )
        elif ctrl:
            delta = self._zoom_last_y - event.y
            self._zoom_last_y = event.y
            if delta:
                self._apply_zoom(1 + delta * 0.01)
        else:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_canvas_release(self, event):
        shift = event.state & 0x1
        if shift and self._rubber_band_start and self._drag_moved:
            if self._rubber_band_id:
                self.canvas.delete(self._rubber_band_id)
                self._rubber_band_id = None
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            x0, y0 = self._rubber_band_start
            rx0 = min(x0, cx) / self.canvas_scale
            ry0 = min(y0, cy) / self.canvas_scale
            rx1 = max(x0, cx) / self.canvas_scale
            ry1 = max(y0, cy) / self.canvas_scale
            for k, (li, wi) in enumerate(self.word_map):
                word = self.raw_data[li]['words'][wi]
                wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                if not wx: continue
                if min(wx) < rx1 and max(wx) > rx0 and min(wy) < ry1 and max(wy) > ry0:
                    self.selected_ks.add(k)
            self._update_selection_status()
            self._redraw_canvas()
        elif not self._drag_moved:
            self._on_canvas_click(event)

    def _apply_zoom(self, factor):
        self._zoom_level = max(0.2, min(5.0, self._zoom_level * factor))
        self._redraw_canvas()

    def _on_zoom_scroll(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self._apply_zoom(factor)

    def _on_canvas_click(self, event):
        ctrl  = event.state & 0x4
        shift = event.state & 0x1
        cx = self.canvas.canvasx(event.x) / self.canvas_scale
        cy = self.canvas.canvasy(event.y) / self.canvas_scale
        hits = []
        for k, (li, wi) in enumerate(self.word_map):
            word = self.raw_data[li]['words'][wi]
            wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
            wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
            x0, y0, x1, y1 = min(wx), min(wy), max(wx), max(wy)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                hits.append(((x1 - x0) * (y1 - y0), k))
        if hits:
            hits.sort()
            k = hits[0][1]
            if ctrl or shift:
                if k in self.selected_ks:
                    self.selected_ks.discard(k)
                else:
                    self.selected_ks.add(k)
                self._update_selection_status()
                self._redraw_canvas()
                return
            # 단일 선택
            self.selected_ks.clear()
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(k)
            self.listbox.see(k)
            li, wi = self.word_map[k]
            word = self.raw_data[li]['words'][wi]
            wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
            wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
            wb = [min(wx) * self.canvas_scale, min(wy) * self.canvas_scale,
                  max(wx) * self.canvas_scale, max(wy) * self.canvas_scale]
            self.canvas.delete("highlight")
            self.canvas.create_rectangle(wb[0], wb[1], wb[2], wb[3],
                                         outline="blue", width=3, tags="highlight")
        else:
            if not (ctrl or shift):
                self.selected_ks.clear()
                self._update_selection_status()
                self._redraw_canvas()

    def on_select_word(self, event):
        sel = self.listbox.curselection()
        if not sel: return
        k = sel[0]
        if k >= len(self.word_map): return
        if event is not None:  # 사용자 클릭 시에만 선택 초기화
            self.selected_ks.clear()

        li, wi = self.word_map[k]
        word = self.raw_data[li]['words'][wi]
        wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
        wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
        wb = [min(wx) * self.canvas_scale, min(wy) * self.canvas_scale,
              max(wx) * self.canvas_scale, max(wy) * self.canvas_scale]

        self.canvas.delete("highlight")
        self.canvas.create_rectangle(wb[0], wb[1], wb[2], wb[3], outline="blue", width=3, tags="highlight")
        self.canvas.yview_moveto(max(0.0, wb[1] / self.preview_tk.height() - 0.1))

    def on_delete_word(self, event):
        sel = self.listbox.curselection()
        if not sel: return
        k = sel[0]
        if k >= len(self.word_map): return

        li, wi = self.word_map[k]
        self.raw_data[li]['words'].pop(wi)
        # 단어가 없어진 줄은 제거
        if not self.raw_data[li]['words']:
            self.raw_data.pop(li)

        self.canvas.delete("highlight")
        self.refresh_ui_data()

    def _update_selection_status(self):
        n = len(self.selected_ks)
        pdf_name = self.pdf_list[self.current_pdf_idx]
        base = f"파일: {pdf_name} | 페이지: {self.current_page_idx + 1} / {len(self.doc)}"
        if n >= 2:
            self.lbl_status.config(text=f"{base}  |  {n}개 선택됨 — Ctrl+G: 병합, Ctrl+Shift+G: 해체")
        elif n == 1:
            self.lbl_status.config(text=f"{base}  |  1개 선택됨 — Ctrl+Shift+G: 해체 (병합된 경우)")
        else:
            self.lbl_status.config(text=base)

    def group_selected(self, event=None):
        if len(self.selected_ks) < 2:
            return

        # 선택된 word들을 읽기 순서(위→아래, 왼→오)로 정렬
        selected_items = []
        for k, (li, wi) in enumerate(self.word_map):
            if k in self.selected_ks:
                word = self.raw_data[li]['words'][wi]
                wx = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                wy = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                selected_items.append((min(wy), min(wx), li, wi, word))
        selected_items.sort()

        all_chars = []
        all_texts = []
        orig_words = []
        all_wx, all_wy = [], []

        for _, _, li, wi, word in selected_items:
            all_chars.extend(word['chars'])
            all_texts.append(word['word'])
            orig_words.append({'word': word['word'], 'chars': list(word['chars']),
                               '_grouped': word.get('_grouped', False),
                               '_orig_words': word.get('_orig_words', [])})
            for c in word['chars']:
                all_wx += [c['bbox'][0], c['bbox'][2]]
                all_wy += [c['bbox'][1], c['bbox'][3]]

        merged_word = {
            'word': ''.join(all_texts),
            'chars': all_chars,
            '_grouped': True,
            '_orig_words': orig_words
        }
        merged_bbox = [min(all_wx), min(all_wy), max(all_wx), max(all_wy)]

        # 선택된 word 제거
        to_remove = {self.word_map[k] for k in self.selected_ks}
        for li_idx, line in enumerate(self.raw_data):
            line['words'] = [w for wi_idx, w in enumerate(line['words'])
                             if (li_idx, wi_idx) not in to_remove]
        self.raw_data = [line for line in self.raw_data if line['words']]

        # 병합 word를 새 라인으로 추가
        self.raw_data.append({
            'text': merged_word['word'],
            'words': [merged_word],
            'line_bbox': merged_bbox
        })

        self.selected_ks.clear()
        self.refresh_ui_data()

    def ungroup_selected(self, event=None):
        if len(self.selected_ks) != 1:
            return
        k = next(iter(self.selected_ks))
        if k >= len(self.word_map):
            return
        li, wi = self.word_map[k]
        word = self.raw_data[li]['words'][wi]
        if not word.get('_grouped'):
            return

        orig_words = word.get('_orig_words', [])

        # 병합 word 제거
        self.raw_data[li]['words'].pop(wi)
        if not self.raw_data[li]['words']:
            self.raw_data.pop(li)

        # 원본 word들을 각각 개별 라인으로 복원
        for orig_word in orig_words:
            self.raw_data.append({
                'text': orig_word['word'],
                'words': [orig_word],
                'line_bbox': []
            })

        self.selected_ks.clear()
        self.refresh_ui_data()

    def save_and_next(self):
        self.history.append((self.current_pdf_idx, self.current_page_idx))
        pdf_name_full = self.pdf_list[self.current_pdf_idx]
        pdf_name = os.path.splitext(pdf_name_full)[0]
        page_name = f"{pdf_name}_p{self.current_page_idx:03d}"
        
        # 1. 이미지 저장 (1280px)
        img_path = os.path.join(IMG_OUT_DIR, f"{page_name}.png")
        self.master_img.save(img_path)
        
        # 2. JSON 저장 (MMOCR 인스턴스 형태)
        instances = []
        for line in self.raw_data:
            for word in line['words']:
                # word 단위 bbox 계산
                w_x = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                w_y = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                wb = [min(w_x), min(w_y), max(w_x), max(w_y)]
                
                instances.append({
                    "polygon": [wb[0], wb[1], wb[2], wb[1], wb[2], wb[3], wb[0], wb[3]],
                    "bbox": wb,
                    "bbox_label": 0,
                    "ignore": False,
                    "text": word['word']
                })
        
        ann_data = {
            "img_path": f"images/{page_name}.png",
            "height": self.master_img.height,
            "width": self.master_img.width,
            "instances": instances
        }
        
        with open(os.path.join(ANN_OUT_DIR, f"{page_name}.json"), "w", encoding="utf-8") as f:
            json.dump(ann_data, f, ensure_ascii=False, indent=2)
            
        print(f"저장 완료: {page_name}")
        self.current_page_idx += 1
        self.load_next_page()

    def save_vertical_and_next(self):
        """세로 텍스트가 포함된 페이지를 별도 폴더(images_vertical / annotations_vertical)에 저장."""
        self.history.append((self.current_pdf_idx, self.current_page_idx))
        pdf_name_full = self.pdf_list[self.current_pdf_idx]
        pdf_name = os.path.splitext(pdf_name_full)[0]
        page_name = f"{pdf_name}_p{self.current_page_idx:03d}"

        # 1. 이미지 저장 (세로 전용 폴더)
        self.master_img.save(os.path.join(IMG_VERT_DIR, f"{page_name}.png"))

        # 2. JSON 저장 (세로 전용 폴더, 형식은 save_and_next와 동일)
        instances = []
        for line in self.raw_data:
            for word in line['words']:
                w_x = [c['bbox'][0] for c in word['chars']] + [c['bbox'][2] for c in word['chars']]
                w_y = [c['bbox'][1] for c in word['chars']] + [c['bbox'][3] for c in word['chars']]
                wb = [min(w_x), min(w_y), max(w_x), max(w_y)]
                instances.append({
                    "polygon": [wb[0], wb[1], wb[2], wb[1], wb[2], wb[3], wb[0], wb[3]],
                    "bbox": wb,
                    "bbox_label": 0,
                    "ignore": False,
                    "text": word['word']
                })

        ann_data = {
            "img_path": f"images_vertical/{page_name}.png",
            "height": self.master_img.height,
            "width": self.master_img.width,
            "instances": instances
        }
        with open(os.path.join(ANN_VERT_DIR, f"{page_name}.json"), "w", encoding="utf-8") as f:
            json.dump(ann_data, f, ensure_ascii=False, indent=2)

        print(f"세로 저장 완료: {page_name}")
        self.current_page_idx += 1
        self.load_next_page()

    def skip_and_next(self):
        self.history.append((self.current_pdf_idx, self.current_page_idx))
        print("페이지 스킵")
        self.current_page_idx += 1
        self.load_next_page()

    def delete_current_page(self):
        pdf_name_full = self.pdf_list[self.current_pdf_idx]
        pdf_name = os.path.splitext(pdf_name_full)[0]
        page_name = f"{pdf_name}_p{self.current_page_idx:03d}"

        targets = [
            os.path.join(IMG_OUT_DIR,  f"{page_name}.png"),
            os.path.join(ANN_OUT_DIR,  f"{page_name}.json"),
            os.path.join(IMG_VERT_DIR, f"{page_name}.png"),
            os.path.join(ANN_VERT_DIR, f"{page_name}.json"),
        ]
        existing = [p for p in targets if os.path.exists(p)]

        if not existing:
            messagebox.showinfo("삭제", f"저장된 파일이 없습니다.\n({page_name})")
            return

        names = "\n".join(os.path.relpath(p, DATASET_ROOT) for p in existing)
        if not messagebox.askyesno("삭제 확인", f"다음 파일을 삭제합니다:\n{names}\n\n계속하시겠습니까?"):
            return

        for p in existing:
            os.remove(p)

        messagebox.showinfo("삭제 완료", f"{len(existing)}개 파일을 삭제했습니다.")

    def go_back(self):
        if not self.history:
            messagebox.showinfo("알림", "돌아갈 이전 페이지가 없습니다.")
            return

        prev_pdf_idx, prev_page_idx = self.history.pop()

        if self.doc is not None:
            self.doc.close()
            self.doc = None

        self.current_pdf_idx = prev_pdf_idx
        self.current_page_idx = prev_page_idx

        pdf_name = self.pdf_list[self.current_pdf_idx]
        raw_path = os.path.join(RAW_PDF_DIR, pdf_name)
        proc_path = os.path.join(PROCESSED_PDF_DIR, pdf_name)

        if os.path.exists(raw_path):
            self.doc = fitz.open(raw_path)
        elif os.path.exists(proc_path):
            self.doc = fitz.open(proc_path)
        else:
            messagebox.showerror("오류", f"PDF 파일을 찾을 수 없습니다: {pdf_name}")
            return

        self.process_page()

    def _complement_color(self, img):
        r, g, b = img.resize((1, 1), Image.LANCZOS).getpixel((0, 0))
        return f"#{255-r:02x}{255-g:02x}{255-b:02x}"

    def export_to_mmocr(self):
        """개별 JSON들을 모아서 최종 textdet_train.json 생성"""
        all_json_files = [f for f in os.listdir(ANN_OUT_DIR) if f.endswith(".json")]
        if not all_json_files:
            messagebox.showwarning("경고", "저장된 데이터가 없습니다.")
            return
            
        data_list = []
        for jf in all_json_files:
            with open(os.path.join(ANN_OUT_DIR, jf), "r", encoding="utf-8") as f:
                data_list.append(json.load(f))
        
        metainfo = {
            "dataset_type": "TextDetDataset",
            "task_name": "textdet",
            "category": [{"id": 0, "name": "text"}]
        }
        
        # 9:1 분리
        import random
        random.shuffle(data_list)
        split_idx = int(len(data_list) * 0.9)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
        
        for name, data in [("train", train_data), ("val", val_data)]:
            out_path = os.path.join(DATASET_ROOT, f"textdet_{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"metainfo": metainfo, "data_list": data}, f, ensure_ascii=False, indent=2)
        
        messagebox.showinfo("성공", f"최종 병합 완료!\nTrain: {len(train_data)}장\nVal: {len(val_data)}장")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1600x900")
    app = IntegratedGUI(root)
    root.mainloop()
