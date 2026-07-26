import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io

class PDFRenderer:
    """
    PDF 페이지를 이미지로 변환하는 렌더러.
    - target_width 지정 시: 페이지 너비를 target_width px로 맞추고 높이는 비율 유지
    - target_width=None 시: dpi 기반 렌더링 (기존 방식)
    """
    def __init__(self, dpi: int = 300, target_width: int | None = None):
        self.dpi = dpi
        self.target_width = target_width
        # target_width 미지정 시에만 고정 zoom 사전 계산
        if target_width is None:
            zoom = dpi / 72.0
            self.matrix = fitz.Matrix(zoom, zoom)
        else:
            self.matrix = None  # 페이지마다 동적 계산

    def render_page(self, pdf_path: str, page_number: int = 0) -> np.ndarray:
        """
        특정 페이지를 numpy 배열(RGB)로 반환.
        """
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_number)

            if self.target_width is not None:
                # 긴 쪽(최대 길이)을 target_width(1280px)에 맞추어 zoom 계산
                max_pt = max(page.rect.width, page.rect.height)
                zoom = self.target_width / max_pt
                matrix = fitz.Matrix(zoom, zoom)
            else:
                matrix = self.matrix

            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            return np.array(image)
        except Exception as e:
            raise ValueError(f"PDF 렌더링 실패: {e}")
        finally:
            doc.close()
