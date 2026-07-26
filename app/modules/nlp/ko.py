"""
한국어 얕은(shallow) 청커 — Kiwi 형태소 품사 태그 기반.

기존 Stanza depparse 방식은 "문장에는 지배 동사가 있다"는 전제로 트리를 만드는데,
PPT 불릿·표·이름+자격 나열 같은 비문장형 파편 텍스트에서는 이 전제가 깨져서
- 특정 deprel이 분류표에 없으면 통째로 유실되고 (예: 'list', 'nmod:unmarked')
- 기호(>, ", /)가 청크 경계로 인식되지 않아 서로 다른 항목이 섞여 붙고
- 서술어(VP)조차 NP로 잘못 라벨링되는 문제가 있었음.

이 구현은 의존구문분석 없이, 어절(공백 기준) 단위로 마지막 형태소 태그를 보고
체언(NP) / 용언(VP) / 관형 수식(MOD) / 경계(BREAK)로 역할만 분류해서 이어붙임.
문장 구조를 요구하지 않으므로 파편 텍스트에도 동일하게 동작하고, 분류 안 되는
어절은 (Stanza 버전처럼 조용히 버리지 않고) NP로 폴백 처리해서 유실을 막음.
"""

from __future__ import annotations

from kiwipiepy import Kiwi

from .chunker import DepparseChunker

# 체언류 — 명사구(NP)를 구성하는 핵심 태그
_NOUN_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'SL', 'SN', 'SH', 'XR'}
# 관형격 조사("~의") — NP 내부 수식으로 계속 이어붙임 (다음 어절과 병합)
_NP_CONTINUE_PARTICLES = {'JKG'}
# 주격/목적격/부사격/보격/호격/인용격/보조사/접속조사 — 하나의 논항이 완결됐다고 보고
# 다음 어절이 NP여도 새 청크로 끊음 (그래야 "미생물은"+"우리 몸에"+"도움을"이
# 하나의 거대한 덩어리로 안 뭉개짐)
_NP_CLOSE_PARTICLES = {'JKS', 'JKO', 'JKB', 'JKC', 'JKV', 'JKQ', 'JX', 'JC'}
_NOUN_PARTICLE_TAGS = _NP_CONTINUE_PARTICLES | _NP_CLOSE_PARTICLES
# 용언 어간
_VERB_TAGS = {'VV', 'VA', 'VX', 'VCP', 'VCN'}
# 용언 종결/연결 어미 — 이 어미로 끝나면 서술어(VP)
_VERB_ENDING_TAGS = {'EP', 'EF', 'EC'}
# 명사형 전성어미(-음/-기) — 용언이 명사처럼 쓰이므로 NP 취급
_NOMINALIZING_ENDING = 'ETN'
# 관형형 전성어미(-ㄴ/-ㄹ/-는) — 다음 어절(NP)을 수식하는 관형 성분
_ADNOMINAL_ENDING = 'ETM'
# 문장부호·특수기호 — 항상 청크 경계 (표/괄호/따옴표/슬래시 등이 서로 다른
# 항목을 이어붙이던 문제를 여기서 원천 차단함)
_BREAK_TAGS = {'SF', 'SP', 'SS', 'SSO', 'SSC', 'SE', 'SO', 'SW'}
# 부사(접속부사 포함) — 핵심어로 보기 애매한 단독 조각이라 경계로 취급
_ADVERB_TAG = 'MAG'


class KoChunker(DepparseChunker):
    """Kiwi 형태소 품사 태그 기반 한국어 얕은 청커."""

    _lang_tag = 'ko'

    def __init__(self, config: dict | None = None):
        self.kiwi = Kiwi()

    def extract_chunks_batch(self, texts: list[str]) -> list[list[tuple[str, str]]]:
        if not texts:
            return []
        results = list(self.kiwi.tokenize(texts))
        return [self._chunk_from_tokens(text, tokens) for text, tokens in zip(texts, results)]

    def extract_chunks(self, text: str) -> list[tuple[str, str]]:
        tokens = self.kiwi.tokenize(text)
        return self._chunk_from_tokens(text, tokens)

    # ── 핵심 로직 ────────────────────────────────────────────────────

    def _chunk_from_tokens(self, text: str, tokens) -> list[tuple[str, str]]:
        words = text.split()
        if not words:
            return []

        # 어절별 문자 오프셋(공백 join 기준) 계산
        word_spans = []
        pos = 0
        for w in words:
            word_spans.append((pos, pos + len(w)))
            pos += len(w) + 1  # +1 공백

        # 각 어절에 속하는 (형태소 태그, 형태) 모음
        word_morphs: list[list[tuple[str, str]]] = [[] for _ in words]
        wi = 0
        for tok in tokens:
            while wi + 1 < len(word_spans) and tok.start >= word_spans[wi][1]:
                wi += 1
            word_morphs[wi].append((tok.tag, tok.form))

        roles = [self._classify_word(morphs) for morphs in word_morphs]
        return self._merge_roles(words, roles)

    @staticmethod
    def _classify_word(morphs: list[tuple[str, str]]) -> tuple[str, bool]:
        """어절의 (형태소 태그, 형태) 목록 → (역할, force_close).

        force_close=True면 이 어절 뒤에서 같은 타입 어절이 이어져도 바로 청크를 끊음
        (예: "미생물은"(주격류) 뒤 "우리 몸에"(부사격류)가 하나로 뭉개지지 않도록).
        """
        if not morphs:
            return 'NP', False
        tags = [t for t, _ in morphs]
        # 어절이 기호로만 이루어진 경우(>, ", / 단독 등) — 항상 경계
        if all(t in _BREAK_TAGS for t in tags):
            return 'BREAK', False

        last_tag, last_form = morphs[-1]
        if last_tag == 'XSN' and last_form == '적':
            # "비교적"/"일반적"처럼 "-적"으로만 끝나는 단독 어절은 보통 뒤따르는
            # 서술어·명사를 꾸미는 수식어로 쓰임 (예: "비교적 빠르게") → MOD로 취급
            return 'MOD', False
        if last_tag == _ADNOMINAL_ENDING:
            return 'MOD', False
        if last_tag == _NOMINALIZING_ENDING:
            return 'NP', False
        if last_tag == 'EF':  # 종결어미 — 서술어가 완결됨
            return 'VP', True
        if last_tag in _VERB_ENDING_TAGS:  # EP/EC — 다음 서술어로 계속 이어짐
            return 'VP', False
        # "있기보다는"(VX+ETN+JKB+JX)처럼 동사 어간 위에 명사형 전성어미(ETN)가
        # 붙고 그 뒤로 조사가 더 붙어 겉보기엔 명사구처럼 끝나는 경우 — 여전히
        # 서술어 의미이므로 NP로 잘못 끊어서 앞의 VP(예: "떠")와 갈라지지 않도록 VP로 봄
        if (
            _NOMINALIZING_ENDING in tags
            and any(t in _VERB_TAGS or t == 'XSV' for t in tags)
            and last_tag in (_NP_CLOSE_PARTICLES | _NP_CONTINUE_PARTICLES)
        ):
            return 'VP', last_tag in _NP_CLOSE_PARTICLES
        if last_tag in _VERB_TAGS:
            return 'VP', False
        if last_tag in _NP_CLOSE_PARTICLES:
            return 'NP', True
        if last_tag in _NP_CONTINUE_PARTICLES or last_tag in _NOUN_TAGS:
            return 'NP', False
        if last_tag == _ADVERB_TAG:
            return 'BREAK', False
        # 분류 불가 태그 — 조용히 버리지 않고 NP로 폴백
        return 'NP', False

    @staticmethod
    def _merge_roles(words: list[str], role_info: list[tuple[str, bool]]) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        mod_buf: list[str] = []
        run_type: str | None = None
        run_words: list[str] = []

        def flush_run():
            nonlocal run_type, run_words
            if run_words:
                chunks.append((run_type, ' '.join(run_words)))
            run_type, run_words = None, []

        def flush_mod():
            nonlocal mod_buf
            if mod_buf:
                chunks.append(('NP', ' '.join(mod_buf)))
                mod_buf = []

        for word, (role, force_close) in zip(words, role_info):
            if role == 'BREAK':
                flush_run()
                flush_mod()
            elif role == 'MOD':
                flush_run()
                mod_buf.append(word)
            else:  # 'NP' | 'VP'
                if run_type == role:
                    run_words.append(word)
                else:
                    flush_run()
                    run_words = mod_buf + [word]
                    mod_buf = []
                    run_type = role
                if force_close:
                    flush_run()

        flush_run()
        flush_mod()
        return chunks
