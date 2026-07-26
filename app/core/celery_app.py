from celery import Celery

celery_app = Celery(
    "pdfmask",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["app.core.tasks"],
)

celery_app.conf.task_track_started = True
# CUDA는 fork 이후(각 워커 프로세스 안)에서만 초기화되어야 하므로,
# 모델은 태스크 최초 실행 시점에 지연 로딩한다 (app/core/tasks.py 참고).
