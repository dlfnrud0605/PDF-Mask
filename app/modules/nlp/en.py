"""
영어 depparse 청커 (Stanza).

청킹 규칙표:
    deprel          → 처리
    ────────────────────────────────────────────────────────────────
    compound        → NP_MOD  "deep learning models"
    amod            → NP_MOD  "spatial features", "efficient processing"
    det             → NP_MOD  "The model", "a dataset"
    nummod          → NP_MOD  "50 documents"
    appos           → NP_MOD  apposition 수식
    acl / acl:relcl → NP_MOD  수식 용언 (단, upos==VERB이면 스킵)
    nmod:poss       → NP_MOD  소유격 "the model's" → "the model"
    nsubj           → NP_HEAD 주어
    nsubj:pass      → NP_HEAD 수동태 주어
    obj / iobj      → NP_HEAD 목적어
    obl             → NP_HEAD 사격어 (전치사구 대상 명사)
    nmod            → NP_HEAD 명사 후치 수식 ("data from sources")
    dislocated      → NP_HEAD 전위 구성
    conj            → VP_HEAD or NP 병렬
    flat            → 직전 NP에 후치 병합 ("John Ying", "United States")
    root            → VP_UPOS→VP / NP_UPOS→NP
    case            → 소유격('s)이면 버퍼 유지, 일반 전치사면 clear
    punct           → 수식어 사이 연결 부호이면 버퍼 유지 (hard-working)
    aux / aux:pass / cop / cc / mark / advmod / dep / parataxis / expl → SKIP

★ 한국어와의 주요 차이:
    nmod  → NP_HEAD (전치사구 목적어)       vs 한국어: NP_MOD (의 수식어)
    flat  → 고유명사 후치 병합               vs 한국어: 없음
    case  → 소유격 판별 후 버퍼 처리         vs 한국어: SKIP
    punct → 구조 기반 하이픈 복원            vs 한국어: SKIP
"""

from __future__ import annotations

import re

import stanza

from .chunker import DepparseChunker

# ── 의존 관계 분류표 ────────────────────────────────────────────────

NP_MOD_RELS  = {
    'compound', 'amod', 'det', 'nummod', 'appos', 'acl', 'acl:relcl',
    'nmod:poss',   # 소유격: "the model's performance" → 'model' 버퍼
    'list',        # 나열형 동격어: "Choi, PhD, RN" 같은 자격/직함 나열
}
NP_HEAD_RELS = {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'nmod', 'dislocated'}
SKIP_RELS    = {
    'aux', 'aux:pass', 'cop', 'cc', 'mark',
    'advmod', 'dep', 'parataxis', 'expl',
}
# case / punct는 별도 처리


def _rel_in(rel: str, rel_set: set[str]) -> bool:
    """deprel 하위분류(예: 'nmod:unmarked')까지 커버하는 소속 확인.

    Stanza는 'nmod:poss'처럼 명시적으로 등록해둔 하위분류 외에도
    'nmod:unmarked' 같은 예상 못한 하위분류를 반환할 수 있음. 정확히
    등록된 문자열이 없으면 콜론 앞 기본 관계(base relation)로 한 번 더 확인해서,
    분류표에 없는 하위분류가 조용히 버려지는 것을 막음.
    """
    if rel in rel_set:
        return True
    base = rel.split(':', 1)[0]
    return base in rel_set

NP_UPOS = {'NOUN', 'PROPN', 'NUM'}
VP_UPOS = {'VERB', 'ADJ'}

# 관계절 헤드 동사를 가리키는 deprel (VERB면 스킵)
ACL_RELS = {'acl', 'acl:relcl'}

# conj 헤드가 술어임을 인정하는 deprel
PRED_HEAD_RELS = {'root', 'conj'}


def _join_tokens(tokens: list[str]) -> str:
    """버퍼 토큰 결합 + 연결 부호 압축.

    예: ['hard', '-', 'working'] → 'hard-working'
        ['3',    '/',  'sec'   ] → '3/sec'
    """
    return re.sub(r'(\w) ([^\w\s]) (\w)', r'\1\2\3', ' '.join(tokens))


class EnChunker(DepparseChunker):
    """영어 Stanza depparse 기반 NP/VP 청커."""

    _lang_tag = 'en'

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}

        verbose = config.get('verbose', False)
        print("[EnChunker] Stanza 영어 파이프라인 로딩 중...")
        self.nlp = stanza.Pipeline(
            'en',
            processors='tokenize,pos,lemma,depparse',
            verbose=verbose,
            tokenize_no_ssplit=True,
        )
        print("[EnChunker] 준비 완료.")

    # ── 핵심 청킹 로직 ──────────────────────────────────────────────

    def _process_sentence(self, sent) -> list[tuple[str, str]]:
        """단일 Stanza 문장 → NP/VP 청크 리스트."""
        chunks: list[tuple[str, str]] = []
        buf: list[str] = []

        for word in sent.words:
            rel  = word.deprel
            upos = word.upos

            if _rel_in(rel, NP_MOD_RELS):
                if rel in ACL_RELS and upos == 'VERB':
                    buf.clear()
                else:
                    buf.append(word.text)

            elif rel == 'conj':
                head_w = sent.words[word.head - 1] if word.head > 0 else None
                head_is_pred = (
                    head_w is not None
                    and head_w.deprel in PRED_HEAD_RELS
                    and head_w.upos in VP_UPOS
                )
                if head_is_pred and upos in VP_UPOS:
                    buf.clear()
                    chunks.append(('VP', word.text))
                elif not head_is_pred:
                    buf.append(word.text)
                    chunks.append(('NP', _join_tokens(buf)))
                    buf = []
                else:
                    buf.append(word.text)

            elif rel == 'flat':
                if chunks and chunks[-1][0] == 'NP':
                    chunks[-1] = ('NP', chunks[-1][1] + ' ' + word.text)
                else:
                    buf.append(word.text)

            elif _rel_in(rel, NP_HEAD_RELS):
                if upos in NP_UPOS:
                    buf.append(word.text)
                    chunks.append(('NP', _join_tokens(buf)))
                    buf = []
                else:
                    buf.clear()

            elif rel == 'root':
                if upos in VP_UPOS:
                    buf.clear()
                    chunks.append(('VP', word.text))
                elif upos in NP_UPOS:
                    buf.append(word.text)
                    chunks.append(('NP', _join_tokens(buf)))
                    buf = []
                else:
                    buf.clear()

            elif rel == 'case':
                head_w = sent.words[word.head - 1] if word.head > 0 else None
                if head_w is None or head_w.deprel != 'nmod:poss':
                    buf.clear()

            elif rel == 'punct':
                # 쉼표 등은 버퍼를 이어주는 연결부호일 수도(hard-working),
                # 목록 구분자("PhD, RN")일 수도 있음 — 어느 쪽인지 애매하면
                # 이미 쌓인 버퍼를 함부로 비우지 않음 (버림보다 보존이 안전함)
                head_w = sent.words[word.head - 1] if word.head > 0 else None
                if head_w is not None and _rel_in(head_w.deprel, NP_MOD_RELS):
                    buf.append(word.text)

            elif _rel_in(rel, SKIP_RELS):
                buf.clear()

            else:
                buf.clear()

        if buf:
            chunks.append(('NP', _join_tokens(buf)))
        return chunks

    def extract_chunks(self, text: str) -> list[tuple[str, str]]:
        """영어 텍스트에서 NP / VP 청크 추출."""
        doc = self.nlp(text)
        chunks: list[tuple[str, str]] = []
        for sent in doc.sentences:
            chunks.extend(self._process_sentence(sent))
        return chunks

    def extract_chunks_batch(self, texts: list[str]) -> list[list[tuple[str, str]]]:
        """여러 영어 텍스트 배치 분석."""
        if not texts:
            return []
        import stanza
        docs = [stanza.Document([], text=t) for t in texts]
        all_chunks = []
        for start in range(0, len(docs), 128):
            processed_docs = self.nlp(docs[start:start + 128])
            for doc in processed_docs:
                chunks = []
                for sent in doc.sentences:
                    chunks.extend(self._process_sentence(sent))
                all_chunks.append(chunks)
        return all_chunks
