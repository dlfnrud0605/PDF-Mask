# PDF Mask

PDF 문서의 레이아웃을 보존한 채 어절 단위로 핵심어를 자동 마스킹하는 학습 자료 생성 시스템.

## 사전 요구사항

- NVIDIA GPU + CUDA (레이아웃 분석/OCR 모델이 `device: cuda`로 설정되어 있음)
- Anaconda/Miniconda
- PyTorch(mmocr, ultralytics 등)와 PaddlePaddle(PaddleOCR)이 GPU 메모리/라이브러리 충돌을 일으키기 때문에, **반드시 두 개의 별도 conda 환경**으로 나눠서 설치해야 함.
- Redis (Celery 브로커/결과 저장소) — `sudo apt install redis-server` 등으로 설치

## 설치

### 1. 메인 환경 (FastAPI 서버 + 레이아웃분석 + DBNet++ + Stanza)

```bash
conda create -n pdfmask_torch python=3.10
conda activate pdfmask_torch
pip install -r requirements/requirements-torch.txt
```

### 2. SVTR 인식 워커 환경 (PaddleOCR 전용, 서브프로세스로 격리 실행됨)

```bash
conda create -n pdfmask_paddle python=3.10
conda activate pdfmask_paddle
pip install -r requirements/requirements-paddle.txt
```

`configs/pipeline.yaml`의 `svtr.conda_env_name` 값(`pdfmask_paddle`)이 위에서 만든 환경 이름과 반드시 일치해야 함.

### 3. Stanza 언어 모델 다운로드 (최초 1회, `pdfmask_torch` 환경에서)

코드에서 자동 다운로드하지 않으므로 미리 받아둬야 함.

```bash
conda activate pdfmask_torch
python -c "import stanza; stanza.download('ko'); stanza.download('en')"
```

## 실행

무거운 파이프라인 처리(레이아웃 분석·OCR·NLP)는 Celery 워커가 백그라운드에서 담당하고, FastAPI는 업로드를 받아 작업을 큐에 등록한 뒤 즉시 응답한다. 그래야 한 요청이 처리되는 동안에도 다른 요청을 받을 수 있음. 아래 세 개를 각각 별도 터미널에서 띄워야 함.

### 1) Redis 실행 (브로커)

```bash
redis-server
```

이미 시스템 서비스로 떠 있다면 생략 가능 (`redis-cli ping` 했을 때 `PONG`이 나오면 실행 중).

### 2) Celery 워커 실행 (실제 GPU 추론 담당)

```bash
conda activate pdfmask_torch
celery -A app.core.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo`를 쓰는 이유: CUDA는 워커 프로세스가 fork된 이후에만 안전하게 초기화할 수 있고, 모델도 GPU 메모리 하나만 쓰므로 동시에 여러 개 로드되면 안 됨. `solo` 풀은 멀티프로세싱/스레딩 없이 태스크를 하나씩 순서대로 처리해서 이 문제를 피함 (동시에 여러 PDF가 들어오면 큐에 쌓였다가 순서대로 처리됨 — 서버가 멈추는 게 아니라 대기열이 생기는 정상적인 동작).

### 3) FastAPI 서버 실행 (요청 접수 전용, 가벼움)

```bash
conda activate pdfmask_torch
uvicorn main:app --host 0.0.0.0 --port 8000
```

기본적으로 `http://127.0.0.1:8000`에서 API 서버가 뜸. API 문서는 `http://127.0.0.1:8000/docs`에서 자동으로 확인 가능함 (FastAPI 기본 제공).

- `POST /api/v1/mask/` — PDF 업로드, 즉시 `{"task_id": "..."}` 반환
- `GET /api/v1/mask/{task_id}` — 처리 상태 확인. 완료되면 마스킹된 PDF 파일을 바로 응답함

## 프론트엔드 사용

`web/index.html`을 브라우저로 그냥 열면 됨 (내부적으로 `http://127.0.0.1:8000/api/v1/mask/`를 호출하도록 되어 있으므로, 위의 서버가 먼저 켜져 있어야 함).

## 모델 가중치

이미 포함되어 있음:
- `models/doclayout_yolo_docstructbench_imgsz1024.pt` — 레이아웃 분석
- `models/yolov8n.pt` — 레이아웃 분석 폴백
- `work_dirs/20260601_032706_400e_lr002/epoch_380.pth` — 자체 파인튜닝한 DBNet++ 텍스트 탐지 가중치

## 설정 변경

`configs/pipeline.yaml`에서 파이프라인 각 단계에 쓸 모델을 교체할 수 있음 (플러그앤플레이 구조).
