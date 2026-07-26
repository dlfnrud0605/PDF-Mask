"""
Stanza depparse 기반 청킹 공통 추상 클래스.

언어별 서브클래스(KoChunker, EnChunker)는
  - self.nlp        : Stanza Pipeline
  - extract_chunks  : 텍스트 → [(chunk_type, text), ...] 구현

infer() 입력/출력 포맷은 KeyBERTExtractor와 동일:
    입력  regions_data: [{"label": ..., "words": [{"text": ..., "bbox": ...}, ...]}]
    출력  [  # 단락 단위
              [{"text": ..., "bbox": ..., "chunk_type": "NP"|"VP"}, ...],
              ...
          ]
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any

from app.core.base import BaseModel
from .spacing import SafeSpacingAligner


class DepparseChunker(BaseModel):
    """언어별 depparse 청커의 공통 추상 베이스."""

    # 청킹 제외 대상 레이아웃 라벨 (Figure, Abandon 등 노이즈 구역 및 제목 영역)
    SKIP_LABELS: frozenset[str] = frozenset({
        "image", "picture", "figure_caption",
        "abandon", "table", "table_caption", "table_footnote", "formula_caption",
        "title"
    })

    # ── 서브클래스가 반드시 구현 ────────────────────────────────────

    @abstractmethod
    def __init__(self, config: dict | None = None):
        """Stanza Pipeline 초기화 등 언어별 준비."""
        ...

    @abstractmethod
    def extract_chunks(self, text: str) -> list[tuple[str, str]]:
        """
        단일 텍스트 → NP/VP 청크 리스트.

        Returns:
            [('NP', '딥러닝 모델이'), ('VP', '향상시킨다'), ...]
        """
        ...

    # ── 공통 infer ──────────────────────────────────────────────────

    def infer(self, regions_data: list[dict]) -> dict[int, list[dict]]:
        """
        단락 리스트를 받아 각 단락의 청킹 마스킹 후보를 반환.

        Args:
            regions_data: YOLO 단락 단위 딕셔너리 리스트
                [
                    {
                        "label": "Text",
                        "region_bbox": [x1, y1, x2, y2],
                        "words": [
                            {"text": "딥러닝", "bbox": [gx1, gy1, gx2, gy2]},
                            ...
                        ]
                    },
                    ...
                ]

        Returns:
            리전 인덱스 → 청크 리스트 매핑.
            Figure·빈 리전처럼 청킹 대상이 아닌 리전은 키 자체가 없음.
            청크 단위로 묶여 있으며, bbox는 구성 어절 전체를 감싸는 합집합 bbox.
                {
                    0: [
                        {
                            "text": "딥러닝 모델이",       # 청크 전체 텍스트
                            "bbox": [x1, y1, x2, y2],    # 청크 합집합 bbox
                            "chunk_type": "NP",           # NP | VP
                            "words": [                    # 구성 어절 목록 (bbox 포함)
                                {"text": "딥러닝", "bbox": [...]},
                                {"text": "모델이", "bbox": [...]},
                            ],
                        },
                        ...
                    ],
                    2: [...],   # 인덱스 1은 Figure라 키 없음
                    ...
                }
        """
        results: dict[int, list[dict]] = {}

        # ── Pass 1: 전처리 — 줄 단위 분리 ──────────────────────────
        # pending: (region_idx, line_words, line_text) — 줄 단위로 쌓음
        pending = []
        for region_idx, region in enumerate(regions_data):
            label = region.get("label", "").lower()
            words: list[dict] = region.get("words", [])

            if label in self.SKIP_LABELS:
                continue

            valid_words = [w for w in words if w.get("text", "").strip()]
            if not valid_words:
                continue

            raw_full_text = " ".join(w["text"] for w in valid_words)

            lang = self._lang_tag
            if lang == 'multi' and hasattr(self, '_detect_lang'):
                lang = self._detect_lang(raw_full_text)

            aligner = SafeSpacingAligner(lang=lang)
            valid_words = aligner.align(valid_words)

            for line_words in self._split_by_line(valid_words):
                line_text = " ".join(w["text"] for w in line_words)
                pending.append((region_idx, line_words, line_text))

        if not pending:
            return results

        # ── Pass 2: 배치 청킹 (Stanza GPU 추론 1회) ─────────────────
        texts = [lt for _, _, lt in pending]
        if hasattr(self, 'extract_chunks_batch'):
            all_chunks = self.extract_chunks_batch(texts)
        else:
            all_chunks = [self.extract_chunks(t) for t in texts]

        # ── Pass 3: 어절 매핑 & bbox 계산 — region별로 합산 ─────────
        region_matched: dict[int, list[dict]] = {}
        for (region_idx, line_words, _), chunks in zip(pending, all_chunks):
            matched = region_matched.setdefault(region_idx, [])
            used: set[int] = set()

            for ctype, form in chunks:
                tokens = form.split()
                chunk_words: list[dict] = []
                cursor = 0

                for token in tokens:
                    for idx in range(cursor, len(line_words)):
                        if idx in used:
                            continue
                        w = line_words[idx]
                        w_text = w["text"]
                        # 1) 정확 일치
                        # 2) 끝 구두점 제거 후 일치  "word." → "word"
                        # 3) 소유격 제거 후 일치      "model's" → "model"
                        #    Stanza는 "model's"를 ["model", "'s"]로 분리하므로
                        #    OCR 워드 "model's"가 청크 토큰 "model"에 매핑되어야 함
                        w_stripped = w_text.rstrip('.,!?。')
                        w_deposs   = w_text.rstrip("'s").rstrip("'s")  # 's / 's 모두 처리
                        if w_text == token or w_stripped == token or w_deposs == token:
                            chunk_words.append(w)
                            used.add(idx)
                            cursor = idx + 1
                            break

                if not chunk_words:
                    continue

                x1 = min(w["bbox"][0] for w in chunk_words)
                y1 = min(w["bbox"][1] for w in chunk_words)
                x2 = max(w["bbox"][2] for w in chunk_words)
                y2 = max(w["bbox"][3] for w in chunk_words)

                matched.append({
                    "text":       form,
                    "bbox":       [x1, y1, x2, y2],
                    "chunk_type": ctype,
                    "words":      chunk_words,
                })

        results = region_matched
        return results

    # ── 내부 헬퍼 ───────────────────────────────────────────────────

    # 가로 간격이 어절 높이의 이 배수를 넘으면, 세로 위치가 같아도 다른 칸(컬럼)으로
    # 보고 줄을 끊음 (예: 나란히 배치된 카드/컬럼의 텍스트가 한 줄로 잘못 이어지는 것 방지)
    _LINE_X_GAP_MAX_RATIO = 3.0

    @staticmethod
    def _split_by_line(words: list[dict]) -> list[list[dict]]:
        """
        Y 좌표 기준으로 어절을 줄 단위로 분리(허용 오차 8px)하되, 가로 간격이 너무
        벌어지면(다른 칸일 가능성) 세로 위치가 같아도 줄을 끊음.
        입력 words는 이미 읽기 순서(줄 단위, 줄 내 왼쪽→오른쪽)로 정렬돼 있다고 가정함.
        """
        if not words:
            return []
        lines: list[list[dict]] = []
        curr: list[dict] = [words[0]]
        curr_y: int = words[0]["bbox"][1]
        curr_x2: float = words[0]["bbox"][2]
        for w in words[1:]:
            y = w["bbox"][1]
            x1 = w["bbox"][0]
            h = w["bbox"][3] - w["bbox"][1]
            x_gap = x1 - curr_x2
            same_y = abs(y - curr_y) <= 8
            same_x = x_gap <= h * DepparseChunker._LINE_X_GAP_MAX_RATIO
            if same_y and same_x:
                curr.append(w)
                curr_x2 = max(curr_x2, w["bbox"][2])
            else:
                lines.append(curr)
                curr = [w]
                curr_y = y
                curr_x2 = w["bbox"][2]
        lines.append(curr)
        return lines

    def infer_pdf(self, all_pages_regions: list[list[dict]]) -> list[dict]:
        """
        여러 페이지 region을 한 번에 받아 Stanza 배치 처리 후 페이지별 결과 반환.
        page당 infer() 개별 호출 대비 Stanza 호출 횟수를 1회로 줄임.
        """
        # ── Pass 1: 전체 페이지 전처리 — 줄 단위 분리 ──────────────
        # (page_idx, region_idx, line_words, line_text)
        all_pending = []
        for page_idx, regions_data in enumerate(all_pages_regions):
            for region_idx, region in enumerate(regions_data):
                label = region.get("label", "").lower()
                words = region.get("words", [])
                if label in self.SKIP_LABELS:
                    continue
                valid_words = [w for w in words if w.get("text", "").strip()]
                if not valid_words:
                    continue
                raw = " ".join(w["text"] for w in valid_words)
                lang = self._lang_tag
                if lang == 'multi' and hasattr(self, '_detect_lang'):
                    lang = self._detect_lang(raw)
                aligner = SafeSpacingAligner(lang=lang)
                valid_words = aligner.align(valid_words)
                for line_words in self._split_by_line(valid_words):
                    line_text = " ".join(w["text"] for w in line_words)
                    all_pending.append((page_idx, region_idx, line_words, line_text))

        if not all_pending:
            return [{} for _ in all_pages_regions]

        # ── Pass 2: 전체 텍스트 한 번에 배치 청킹 ──────────────────
        texts = [lt for _, _, _, lt in all_pending]
        if hasattr(self, 'extract_chunks_batch'):
            all_chunks = self.extract_chunks_batch(texts)
        else:
            all_chunks = [self.extract_chunks(t) for t in texts]

        # ── Pass 3: 페이지별로 결과 조립 — region별 합산 ────────────
        page_results: list[dict] = [{} for _ in all_pages_regions]
        for (page_idx, region_idx, line_words, _), chunks in zip(all_pending, all_chunks):
            matched = page_results[page_idx].setdefault(region_idx, [])
            used: set[int] = set()
            for ctype, form in chunks:
                tokens = form.split()
                chunk_words: list[dict] = []
                cursor = 0
                for token in tokens:
                    for idx in range(cursor, len(line_words)):
                        if idx in used:
                            continue
                        w = line_words[idx]
                        wt = w["text"]
                        if wt == token or wt.rstrip('.,!?。') == token or wt.rstrip("'s") == token:
                            chunk_words.append(w)
                            used.add(idx)
                            cursor = idx + 1
                            break
                if not chunk_words:
                    continue
                x1 = min(w["bbox"][0] for w in chunk_words)
                y1 = min(w["bbox"][1] for w in chunk_words)
                x2 = max(w["bbox"][2] for w in chunk_words)
                y2 = max(w["bbox"][3] for w in chunk_words)
                matched.append({
                    "text": form, "bbox": [x1, y1, x2, y2],
                    "chunk_type": ctype, "words": chunk_words,
                })

        return page_results

    @property
    def _lang_tag(self) -> str:
        """로그용 언어 태그. 서브클래스에서 오버라이드 가능."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
