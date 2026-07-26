"""
SVTR-Tiny 인식 전용 워커.
DBNet++이 잘라낸 어절 이미지(이미 크롭된 상태)만 받으므로 det=False로 인식만 수행.
PaddlePaddle/PyTorch VRAM 충돌 방지를 위해 pdfmask_paddle 콘다 환경에서 단독 실행됨.

persistent 모드: stdin에서 이미지 디렉토리 경로를 한 줄씩 읽어 처리.
"__EXIT__" 수신 시 종료.
"""
import sys
import os
import json
import logging
import traceback
import cv2

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# PaddlePaddle 기본값: 가용 VRAM의 92%를 초기화 시 선점 → 메인 프로세스(YOLO+DBNet)와 충돌
# auto_growth: 실제 필요한 만큼만 동적 할당 (env var 방식)
os.environ["FLAGS_allocator_strategy"] = "auto_growth"


def run_ocr():
    # ppocr 관련 모든 로거 억제
    for name in ["ppocr", "ppocr.utils", "ppocr.postprocess", "paddle", "paddle.fluid"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        from paddleocr import PaddleOCR

        # rec_batch_num: 내부 배치 크기 축소 → cuDNN workspace 비례 감소
        # 64 기준 128MB workspace → 8 기준 ~16MB workspace
        # 속도 영향: SVTR-Tiny는 소형 모델이라 배치 크기 영향 미미
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang="korean",
            show_log=False,
            rec_batch_num=8,
        )
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"모델 로딩 실패: {e}"}), flush=True)
        sys.exit(1)

    # 준비 완료 신호
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__EXIT__":
            break

        try:
            if os.path.isdir(line):
                paths = sorted(
                    os.path.join(line, f)
                    for f in os.listdir(line)
                    if f.endswith(".jpg")
                )
            else:
                paths = [line]

            # det=False 모드는 numpy 배열을 기대함 — 파일 경로 직접 전달 불가
            imgs = [cv2.imread(p) for p in paths]
            imgs = [img for img in imgs if img is not None]
            if not imgs:
                print(json.dumps({"status": "success", "data": []}, ensure_ascii=False), flush=True)
                continue

            # ── 핵심 주의 ──────────────────────────────────────────────────────────────
            # ocr.ocr(list_of_imgs, det=False) 는 내부에서 imgs = [img] 로 한 번 더
            # 감싸기 때문에 배치 전체가 단 1개의 결과 항목으로 뭉쳐져 반환됨.
            # 예) ocr.ocr([img1,img2,img3], det=False) → [[(t1,c1),(t2,c2),(t3,c3)]]
            #                                              ^^^^^ 1개의 외부 항목!
            # 올바른 해결책: text_recognizer()를 직접 호출.
            # text_recognizer(chunk) → ([(t1,c1), (t2,c2), ..., (tN,cN)], elapsed)
            # 이미지 1장당 정확히 1개의 (text, conf) 튜플을 반환함.
            # ──────────────────────────────────────────────────────────────────────────
            import paddle
            BATCH_SIZE = 32
            results_batch = []
            for start in range(0, len(imgs), BATCH_SIZE):
                chunk = imgs[start:start + BATCH_SIZE]
                try:
                    # text_recognizer 직접 호출: 배치 단위로 GPU 처리, 이미지당 1결과 반환
                    rec_res, _ = ocr.text_recognizer(chunk)
                    for item in rec_res:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            text, conf = item
                            results_batch.append([{"text": str(text), "conf": float(conf)}])
                        else:
                            results_batch.append([])
                except Exception as e_inner:
                    # fallback: 이미지 1장씩 처리 (text_recognizer API 미지원 시)
                    print(f"[SVTR 워커] text_recognizer 직접 호출 실패({e_inner}), 1장씩 처리로 전환", file=sys.stderr)
                    for single_img in chunk:
                        try:
                            res = ocr.ocr(single_img, det=False, cls=False)
                            # res = [[('text', conf)]] 형태 (단일 이미지 반환 포맷)
                            texts = []
                            if res and res[0]:
                                for sub in res[0]:
                                    if isinstance(sub, (list, tuple)) and len(sub) == 2:
                                        t, c = sub
                                        texts.append({"text": str(t), "conf": float(c)})
                            results_batch.append(texts)
                        except Exception:
                            results_batch.append([])
                # 정상/fallback 경로 모두 배치 후 캐시 해제 (누적 OOM 방지)
                paddle.device.cuda.empty_cache()

            print(json.dumps({"status": "success", "data": results_batch}, ensure_ascii=False), flush=True)

        except Exception as e:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            print(json.dumps({"status": "error", "message": str(e) or repr(e)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_ocr()
