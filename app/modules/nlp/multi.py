"""
다국어 depparse 청커 (langdetect 기반 자동 언어 감지).

리전 단위로 언어를 감지하여 KoChunker / EnChunker에 자동 분기.

감지 우선순위:
    1. langdetect   — 텍스트가 충분히 긴 경우 (공백 제외 15자 이상)
    2. 한글 비율 휴리스틱 — 짧은 텍스트 폴백 (한글 > 15% → ko)
    3. 'ko' 기본값  — 감지 실패 시

langdetect 비결정성 주의:
    기본적으로 동일 텍스트도 실행마다 결과가 달라질 수 있음.
    DetectorFactory.seed = 0 으로 고정.
"""

from __future__ import annotations

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from .chunker import DepparseChunker
from .ko import KoChunker
from .en import EnChunker

# 재현 가능한 결과 보장
DetectorFactory.seed = 0

# langdetect를 신뢰할 최소 문자 수 (공백 제외)
_MIN_CHARS = 15

# 지원 언어 코드 → 미지원 언어(ja, zh 등)는 ko 폴백
_SUPPORTED = frozenset({'ko', 'en'})


class MultiLangChunker(DepparseChunker):
    """
    langdetect 기반 자동 언어 감지 + KoChunker / EnChunker 라우터.

    파이프라인에서 단일 청커로 등록해두면 한국어·영어 리전을 자동으로 처리함.

    config 예시:
        {
            "ko": {"verbose": False},  # KoChunker 전용 설정
            "en": {"verbose": False},  # EnChunker 전용 설정
        }
        # 또는 공통 설정만 넘기면 ko/en 모두 동일하게 적용
        {"verbose": False}
    """

    _lang_tag = 'multi'

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        ko_cfg = config.get('ko', config)
        en_cfg = config.get('en', config)
        self.ko = KoChunker(ko_cfg)
        self.en = EnChunker(en_cfg)

    # ── 언어 감지 ────────────────────────────────────────────────────

    def _detect_lang(self, text: str) -> str:
        """
        리전 텍스트의 언어 코드를 반환 ('ko' | 'en').

        - 공백 제외 15자 이상: langdetect 사용
        - 미달 또는 감지 실패: 한글 유니코드 비율 휴리스틱
        - 비지원 언어(ja, zh 등): 'ko' 폴백
        """
        stripped = text.replace(' ', '')

        if len(stripped) >= _MIN_CHARS:
            try:
                lang = detect(text)
                if lang in _SUPPORTED:
                    return lang
                # 비지원 언어(es, ja, zh 등) → 아래 휴리스틱으로 진행
                # 예: "Stanza"가 스페인어로 오감지될 때 한글 비율로 재판단
            except LangDetectException:
                pass  # 아래 휴리스틱으로 진행

        # 폴백: 한글 유니코드(AC00–D7A3) 비율
        ko_chars = sum(1 for c in stripped if '가' <= c <= '힣')
        ratio = ko_chars / max(len(stripped), 1)
        return 'ko' if ratio > 0.15 else 'en'

    # ── 청킹 ─────────────────────────────────────────────────────────

    def extract_chunks(self, text: str) -> list[tuple[str, str]]:
        """감지된 언어에 따라 KoChunker 또는 EnChunker로 위임."""
        lang = self._detect_lang(text)
        chunker = self.ko if lang == 'ko' else self.en
        return chunker.extract_chunks(text)

    def extract_chunks_batch(self, texts: list[str]) -> list[list[tuple[str, str]]]:
        """언어별로 그룹핑 후 각 청커에 배치 위임, 원래 순서로 복원."""
        langs = [self._detect_lang(t) for t in texts]

        ko_indices = [i for i, l in enumerate(langs) if l == 'ko']
        en_indices = [i for i, l in enumerate(langs) if l != 'ko']

        all_results: list[list[tuple[str, str]]] = [[] for _ in texts]

        if ko_indices:
            ko_texts = [texts[i] for i in ko_indices]
            ko_results = self.ko.extract_chunks_batch(ko_texts)
            for i, res in zip(ko_indices, ko_results):
                all_results[i] = res

        if en_indices:
            en_texts = [texts[i] for i in en_indices]
            en_results = self.en.extract_chunks_batch(en_texts)
            for i, res in zip(en_indices, en_results):
                all_results[i] = res

        return all_results
