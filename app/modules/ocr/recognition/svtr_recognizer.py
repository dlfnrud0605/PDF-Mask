import os
import json
import cv2
import tempfile
import subprocess
import shutil
import threading
import atexit
import signal
from typing import List, Any

from app.core.base import BaseModel


class SVTRRecognizer(BaseModel):
    """
    PaddleOCR SVTR-Tiny 기반 텍스트 인식 모듈.
    DBNet++이 잘라낸 어절 이미지를 받아 텍스트와 신뢰도를 반환함.
    프레임워크 VRAM 충돌 방지를 위해 pdfmask_paddle 환경의 워커를 서브프로세스로 실행.
    워커는 최초 1회만 시작되어 모델을 유지한 채 stdin/stdout으로 통신함.
    이미지는 temp 파일로 전달 (파이프 버퍼 한계 우회).
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}

        self.drop_threshold = config.get("drop_threshold", 0.6)
        self.conda_env_name = config.get("conda_env_name", "pdfmask_paddle")
        self.torch_env_name = config.get("torch_env_name", "pdfmask_torch")

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
        )
        self.worker_script = os.path.join(base_dir, "scripts", "worker_svtr_ocr.py")

        if not os.path.exists(self.worker_script):
            print(f"[강력 경고] SVTR 워커 스크립트를 찾을 수 없습니다: {self.worker_script}")

        self._process = None
        self._start_worker()

    def _find_conda_env_prefix(self, env_name: str) -> str | None:
        """conda 환경 이름으로 실제 경로(prefix)를 반환. 못 찾으면 None."""
        # 1) CONDA_PREFIX 형제 디렉토리 탐색 (현재 활성 환경 기준)
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            candidate = os.path.join(os.path.dirname(conda_prefix), env_name)
            if os.path.isdir(candidate):
                return candidate

        # 2) 일반적인 conda 설치 위치 탐색
        for base in [
            os.path.expanduser("~/miniconda3/envs"),
            os.path.expanduser("~/anaconda3/envs"),
            os.path.expanduser("~/mambaforge/envs"),
            os.path.expanduser("~/miniforge3/envs"),
            os.path.expanduser("~/.conda/envs"),
        ]:
            candidate = os.path.join(base, env_name)
            if os.path.isdir(candidate):
                return candidate
        return None

    def _get_python_bin(self) -> str:
        prefix = self._find_conda_env_prefix(self.conda_env_name)
        if prefix:
            python_bin = os.path.join(prefix, "bin", "python")
            if os.path.exists(python_bin):
                return python_bin
        return "python"

    def _get_cudnn_env(self) -> dict:
        """torch 환경의 CUDA 라이브러리 경로를 LD_LIBRARY_PATH에 추가한 환경변수 반환.

        paddle 환경에는 cuDNN/cuBLAS 등이 없으므로 torch 환경에서 빌려옴.
        - paddle_prefix/lib: libcudnn.so 등 버전 없는 심볼릭 링크 모음 (사전 설정 필요)
        - torch_prefix/.../torch/lib: 실제 CUDA 라이브러리 파일
        """
        import glob
        env = os.environ.copy()

        cuda_paths = []

        # paddle env lib (버전 없는 symlink 모음)
        paddle_prefix = self._find_conda_env_prefix(self.conda_env_name)
        if paddle_prefix:
            cuda_paths.append(os.path.join(paddle_prefix, "lib"))

        # torch env: torch/lib + nvidia/*/lib
        torch_prefix = self._find_conda_env_prefix(self.torch_env_name)
        if torch_prefix:
            # Python 버전 자동 감지
            py_dirs = sorted(glob.glob(os.path.join(torch_prefix, "lib", "python3.*")), reverse=True)
            for py_dir in py_dirs:
                torch_lib = os.path.join(py_dir, "site-packages", "torch", "lib")
                if os.path.isdir(torch_lib):
                    cuda_paths.append(torch_lib)
                    break  # 첫 번째 발견된 Python 버전 사용

        existing = [p for p in cuda_paths if os.path.isdir(p)]
        if existing:
            prepend = ":".join(existing)
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{prepend}:{old_ld}" if old_ld else prepend
        return env

    def _kill_worker(self):
        """워커 프로세스 강제 종료 (atexit/signal 핸들러용)."""
        if self._process and self._process.poll() is None:
            try:
                self._process.kill()
                self._process.wait(timeout=3)
            except Exception:
                pass

    def _start_worker(self):
        python_bin = self._get_python_bin()
        self._process = subprocess.Popen(
            [python_bin, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._get_cudnn_env(),
        )
        # 메인 프로세스가 어떤 방식으로 종료되어도 워커 서브프로세스 확실히 제거
        # (Ctrl+C 등 KeyboardInterrupt 시 __del__ 미보장 문제 방어)
        atexit.register(self._kill_worker)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                old = signal.getsignal(sig)
                def _handler(signum, frame, _old=old):
                    self._kill_worker()
                    if callable(_old):
                        _old(signum, frame)
                    else:
                        raise SystemExit(0)
                signal.signal(sig, _handler)
            except ValueError:
                # 스레드 내부에서 실행될 경우 signal 등록이 불가능하므로 무시함. (atexit과 __del__이 대신 처리)
                pass

        # 워커 stderr를 실시간으로 출력하는 데몬 스레드
        def _stderr_relay():
            for line in self._process.stderr:
                line = line.rstrip()
                if line:
                    print(f"[SVTR 워커 stderr] {line}")
        threading.Thread(target=_stderr_relay, daemon=True).start()

        # 워커가 {"status": "ready"}를 보낼 때까지 대기
        # PaddleOCR가 init 중에도 GPU 경고를 stdout으로 출력하므로 non-JSON 줄은 건너뜀
        for _ in range(50):
            ready_line = self._process.stdout.readline()
            if not ready_line:
                print("[SVTRRecognizer 오류] 워커가 ready 신호 없이 종료됨")
                return
            ready_line = ready_line.strip()
            if not ready_line:
                continue
            try:
                msg = json.loads(ready_line)
                if msg.get("status") == "ready":
                    print("[SVTRRecognizer] 워커 준비 완료.")
                else:
                    print(f"[SVTRRecognizer 경고] 초기 메시지: {msg}")
                return
            except json.JSONDecodeError:
                print(f"[SVTRRecognizer] 초기화 non-JSON 무시: {ready_line[:80]!r}")

    def _send(self, path: str) -> dict:
        self._process.stdin.write(path + "\n")
        self._process.stdin.flush()
        # PaddleOCR가 경고/로그를 stdout에 섞어 출력할 수 있으므로
        # JSON 줄을 찾을 때까지 non-JSON 줄은 건너뜀 (최대 30줄)
        for _ in range(30):
            line = self._process.stdout.readline()
            if not line:
                return {"status": "error", "message": "워커 응답 없음 (EOF)"}
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                print(f"[SVTRRecognizer] 워커 non-JSON 출력 무시: {line[:120]!r}")
        return {"status": "error", "message": "유효한 JSON 응답을 받지 못함"}

    def infer(self, image_or_images) -> List[Any]:
        """
        단일 NumPy 이미지 또는 이미지 리스트를 받아 인식 결과를 반환.
        - 단일 입력: List[Dict]  예) [{"text": "안녕", "conf": 0.97}]
        - 배치 입력: List[List[Dict]]
        """
        is_batch = isinstance(image_or_images, list)
        images = image_or_images if is_batch else [image_or_images]

        valid_indices = [
            i for i, img in enumerate(images)
            if img is not None and img.size > 0
            and img.shape[0] >= 5 and img.shape[1] >= 5
        ]
        valid_images = [images[i] for i in valid_indices]

        if not valid_images:
            return [] if not is_batch else [[] for _ in images]

        # /dev/shm = tmpfs (RAM 디스크) → SSD I/O 없이 메모리에서 읽고 씀
        # 없으면 (Windows 등) 일반 /tmp 로 fallback
        shm_base = "/dev/shm" if os.path.isdir("/dev/shm") else None
        temp_dir = tempfile.mkdtemp(prefix="svtr_batch_", dir=shm_base)
        try:
            for i, img in enumerate(valid_images):
                cv2.imwrite(os.path.join(temp_dir, f"{i:04d}.jpg"), img)

            res_dict = self._send(temp_dir)

            if res_dict.get("status") != "success":
                print(f"[SVTRRecognizer 워커 내부 에러] {res_dict.get('message', '알 수 없는 에러')}")
                return [] if not is_batch else [[] for _ in images]

            all_valid_texts = []
            for raw_texts in res_dict.get("data", []):
                all_valid_texts.append(
                    [item for item in raw_texts if item["conf"] >= self.drop_threshold]
                )

        except Exception as e:
            print(f"[SVTRRecognizer 에러] {e}")
            all_valid_texts = []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # 원본 이미지 순서에 맞춰 결과 복원
        final_results = [[] for _ in images]
        for result_idx, original_idx in enumerate(valid_indices):
            final_results[original_idx] = (
                all_valid_texts[result_idx] if result_idx < len(all_valid_texts) else []
            )

        return final_results if is_batch else final_results[0]

    def __del__(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write("__EXIT__\n")
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
