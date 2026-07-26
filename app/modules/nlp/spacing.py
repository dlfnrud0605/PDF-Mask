from __future__ import annotations
import kss
from symspellpy import SymSpell
import pkg_resources

class SafeSpacingAligner:
    """
    OCR 오인식으로 인해 줄바꿈에서 잘못 쪼개진 단어들(예: "안녕하\n세요")을 이어붙이는 교정기.
    파이프라인에서 Y좌표가 정규화된 BBox를 입력받아, Y값이 다를 때(줄바꿈)만 공백을 제거하고 교정합니다.
    """
    def __init__(self, lang: str = 'ko'):
        self.lang = lang
        if self.lang == 'ko':
            pass # kss는 모듈 레벨에서 사용
        else:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
    def align(self, words: list[dict]) -> list[dict]:
        if not words:
            return []
            
        # 1. 원본 텍스트 조합 (줄바꿈 타겟팅)
        orig_text = ""
        has_line_break = False
        for i, w in enumerate(words):
            if i == 0:
                orig_text += w["text"]
            else:
                is_line_break = words[i-1]["bbox"][1] != w["bbox"][1]
                if is_line_break:
                    has_line_break = True
                    orig_text += w["text"]
                else:
                    orig_text += " " + w["text"]

        # 줄바꿈이 없으면 kss 스킵 — 교정할 붙여쓰기가 없음
        if not has_line_break:
            return words

        # 2. 교정기에 통과
        if self.lang == 'ko':
            try:
                corrected_text = kss.correct_spacing(orig_text)
            except Exception as e:
                print(f"  [Spacing] 교정 실패: {e}")
                return words
        else:
            try:
                corrected_text = self.sym_spell.word_segmentation(orig_text).corrected_string
            except Exception as e:
                print(f"  [Spacing] 영문 교정 실패: {e}")
                return words

        # 교정 전후가 같으면 낭비 없이 바로 반환
        if orig_text == corrected_text:
            return words
            
            
        corrected_tokens = corrected_text.split()
        
        # 3. Greedy Join Alignment (안전 최우선 동기화 정렬)
        aligned_words = []
        orig_idx = 0
        n_orig = len(words)
        
        for c_token in corrected_tokens:
            if orig_idx >= n_orig:
                # 교정된 토큰은 남았는데 원본 단어를 다 썼다면 (교정기가 단어를 창조함) -> 즉시 취소
                return words
                
            current_join = ""
            start_idx = orig_idx
            match_found = False
            
            # 현재 교정 토큰(c_token)을 원본 단어들의 연속적 결합으로 만들 수 있는지 확인
            for i in range(orig_idx, n_orig):
                current_join += words[i]["text"]
                if current_join == c_token:
                    # 완벽히 일치 (성공적으로 여러 단어가 하나로 합쳐짐)
                    match_found = True
                    
                    # 합쳐진 단어들의 원본 Bbox들을 모두 취합하여 거대한 박스(Union) 생성
                    merged_bbox = list(words[start_idx]["bbox"])
                    for j in range(start_idx + 1, i + 1):
                        box = words[j]["bbox"]
                        merged_bbox[0] = min(merged_bbox[0], box[0])
                        merged_bbox[1] = min(merged_bbox[1], box[1])
                        merged_bbox[2] = max(merged_bbox[2], box[2])
                        merged_bbox[3] = max(merged_bbox[3], box[3])
                    
                    aligned_words.append({
                        "text": current_join,
                        "bbox": merged_bbox
                    })
                    orig_idx = i + 1
                    break
                elif len(current_join) > len(c_token):
                    # 결합된 길이가 교정 토큰보다 길어짐 
                    # -> 교정기가 새로운 띄어쓰기를 추가하여 단어를 분리했거나 글자를 수정함!
                    # 예: 원본 "아버지방에" -> 교정 "아버지"
                    # "붙이는 기능만 사용"이라는 원칙에 위배되므로 여기서 즉시 중단.
                    break
                    
            if not match_found:
                # 단 하나라도 아다리가 맞지 않으면 쿨하게 전체 교정을 취소하고 무결점 원본 반환
                return words
                
        # 교정 토큰 검사를 다 마쳤는데 원본 단어가 찌꺼기처럼 남아있다면 (교정기가 특수문자 등을 삭제함)
        if orig_idx < n_orig:
            return words
            
        print(f"  [Spacing] 붙여쓰기 교정 완료: {n_orig}어절 -> {len(aligned_words)}어절 안전 병합 성공")
        return aligned_words
