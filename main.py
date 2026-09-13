import json
import os
import shutil
import time
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.celery_app import celery_app
from app.core.pipeline import ANALYZE_IMAGE_TMP_DIR
from app.core.tasks import analyze_pdf_task

app = FastAPI(title="PDF Mask API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


ANALYZE_IMAGE_MAX_AGE_SECONDS = 24 * 3600  # 하루 지나도 확정 안 된 분석 결과 이미지는 정리함


def _cleanup_stale_analyze_images(max_age_seconds: int = ANALYZE_IMAGE_MAX_AGE_SECONDS):
    """
    analyze만 하고 확정(/api/v1/corrections/)까지 안 간 세션은 ANALYZE_IMAGE_TMP_DIR에
    이미지가 계속 남게 됨. 별도 스케줄러 없이, 새 analyze 요청이 들어올 때마다 오래된
    폴더를 가볍게 정리함(요청에 얹혀서 처리 — opportunistic cleanup).
    """
    if not os.path.isdir(ANALYZE_IMAGE_TMP_DIR):
        return
    now = time.time()
    for name in os.listdir(ANALYZE_IMAGE_TMP_DIR):
        path = os.path.join(ANALYZE_IMAGE_TMP_DIR, name)
        try:
            if now - os.path.getmtime(path) > max_age_seconds:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


@app.post("/api/v1/analyze/")
async def analyze_pdf(
    file: UploadFile = File(...),
    mode: str = Form("digital"),
):
    """
    실제 마스킹 PDF를 만들지 않고, 페이지별 원본 어절 위치(bbox, pt 단위)와 점수만 반환.
    청킹(NLP 병합) 없이 어절 그대로 주고, 프론트에서 사용자가 직접 묶어서 청킹함.
    프론트가 원본 PDF를 직접 렌더링(예: PDF.js)하고, 이 위치값을 그 위에 오버레이하는 용도.
    """
    if mode not in ["ocr", "digital"]:
        return JSONResponse({"error": "Invalid mode. Must be 'ocr' or 'digital'."}, status_code=400)

    _cleanup_stale_analyze_images()

    os.makedirs("data/input/tmp", exist_ok=True)

    file_id = str(uuid.uuid4())
    input_path = f"data/input/tmp/{file_id}_{file.filename}"

    with open(input_path, "wb") as destination:
        destination.write(await file.read())

    # file_id는 나중에(확정 시점에) 이 분석에서 이미 렌더링해 둔 페이지 이미지
    # (ANALYZE_IMAGE_TMP_DIR/{file_id}/)를 찾는 참조값으로 프론트에 그대로 돌려줌.
    # 원본 업로드 파일 자체는 analyze가 끝나면 바로 지워짐(analyze_pdf_task 참고).
    task = analyze_pdf_task.delay(input_path, mode, file_id)
    return {"task_id": task.id}


@app.get("/api/v1/analyze/{task_id}")
async def get_analyze_result(task_id: str):
    task = celery_app.AsyncResult(task_id)

    if task.state == "PENDING":
        return JSONResponse({"status": "pending"})
    elif task.state == "STARTED":
        return JSONResponse({"status": "processing"})
    elif task.state == "FAILURE":
        return JSONResponse({"status": "failed", "error": str(task.result)}, status_code=500)
    elif task.state == "SUCCESS":
        return JSONResponse({"status": "done", **task.result})
    else:
        return JSONResponse({"status": task.state.lower()})


class WordItem(BaseModel):
    word_id: str
    text: str
    bbox: list[float]


class PageCorrection(BaseModel):
    page_num: int
    width: float
    height: float
    words: list[WordItem]
    final_groups: list[list[str]]


class CorrectionSubmission(BaseModel):
    mode: str
    mask_ratio: float
    pages: list[PageCorrection]
    file_id: str | None = None  # /api/v1/analyze/ 응답에서 받은 값을 그대로 돌려보냄


CORRECTIONS_PATH = "data/corrections/corrections.jsonl"
CORRECTIONS_IMAGE_DIR = "data/corrections/images"


def _find_analyze_image_dir(file_id: str) -> str | None:
    """
    file_id(uuid)로 analyze 단계에서 이미 렌더링해 둔 페이지 이미지 폴더를 찾음.
    클라이언트가 보낸 값은 uuid 형식인지만 검증하고, 실제 경로는 서버가 통제하는 고정
    디렉터리(ANALYZE_IMAGE_TMP_DIR) 안에서만 찾음 — 경로 조작(path traversal)으로
    임의 경로를 참조하지 못하게 하기 위함.
    """
    try:
        uuid.UUID(file_id)
    except (ValueError, TypeError, AttributeError):
        return None
    path = os.path.join(ANALYZE_IMAGE_TMP_DIR, file_id)
    return path if os.path.isdir(path) else None


@app.post("/api/v1/corrections/")
async def submit_correction(payload: CorrectionSubmission):
    """
    사용자가 화면에서 직접 고친 최종 청킹(어절을 어떻게 묶었는지)을 학습 데이터로 저장함.
    원본 어절(words)과 최종 그룹(final_groups)을 그대로 남겨서, 나중에 청킹 모델
    재학습 시 "정답 청킹"으로 바로 쓸 수 있게 함.

    페이지 이미지는 LayoutLM류 텍스트+위치+이미지 모델을 염두에 두고 같이 저장함.
    새로 렌더링하지 않고, analyze 단계에서 OCR/스케일 계산용으로 이미 만들어 둔
    이미지(ANALYZE_IMAGE_TMP_DIR/{file_id}/)를 그대로 재사용 — 그 폴더를 통째로
    정답 데이터 보관 폴더로 옮기기만 함(폴더 이동이라 사실상 즉시 끝남, 추가 렌더링
    비용 없음). 확정 안 된 채 오래 남은 폴더는 다음 analyze 요청 때 자동 정리됨.
    """
    os.makedirs(os.path.dirname(CORRECTIONS_PATH), exist_ok=True)
    os.makedirs(CORRECTIONS_IMAGE_DIR, exist_ok=True)

    correction_id = str(uuid.uuid4())
    src_image_dir = _find_analyze_image_dir(payload.file_id) if payload.file_id else None

    image_dir = None
    if src_image_dir:
        image_dir = f"{CORRECTIONS_IMAGE_DIR}/{correction_id}"
        shutil.move(src_image_dir, image_dir)

    record = {
        "correction_id": correction_id,
        "created_at": time.time(),
        "mode": payload.mode,
        "mask_ratio": payload.mask_ratio,
        "image_dir": image_dir,  # 각 페이지 이미지는 이 폴더 안 p{page_num}.jpg
        "pages": [page.model_dump() for page in payload.pages],
    }
    with open(CORRECTIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "saved", "correction_id": correction_id}
