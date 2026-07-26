import os

from app.core.celery_app import celery_app
from app.core.pipeline import PDFMaskingPipeline

# 워커 프로세스당 한 번만 로드 (모델이 GPU에 상주)
_pipelines: dict[str, PDFMaskingPipeline] = {}


def _get_pipeline(mode: str) -> PDFMaskingPipeline:
    if mode not in _pipelines:
        _pipelines[mode] = PDFMaskingPipeline(mode=mode)
    return _pipelines[mode]


@celery_app.task(name="mask_pdf_task", bind=True)
def mask_pdf_task(self, input_path: str, output_path: str, mode: str, mask_ratio: float):
    try:
        pipeline = _get_pipeline(mode)
        pipeline.process(input_pdf_path=input_path, output_pdf_path=output_path, mask_ratio=mask_ratio)
        return {"output_path": output_path}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@celery_app.task(name="analyze_pdf_task", bind=True)
def analyze_pdf_task(self, input_path: str, mode: str, file_id: str):
    """
    분석만 하고 마스킹 PDF는 안 만듦. analyze() 안에서 이미 OCR/스케일 계산용으로
    렌더링한 페이지 이미지를 file_id 기준으로 임시 보관해 두므로(ANALYZE_IMAGE_TMP_DIR
    참고), 원본 업로드 파일 자체는 여기서 바로 지워도 됨 — 나중에 확정 시점에 이 파일을
    다시 열 필요가 없음.
    """
    try:
        pipeline = _get_pipeline(mode)
        return {"pages": pipeline.analyze(input_pdf_path=input_path, file_id=file_id), "file_id": file_id}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
