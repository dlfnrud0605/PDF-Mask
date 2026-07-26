from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    모든 AI 모듈(Layout, OCR, NLP 등)이 상속받아야 하는 최상위 추상 클래스입니다.
    이 규격을 지켜야만 나중에 모델을 자유롭게 갈아끼울 수 있습니다.
    """

    @abstractmethod
    def __init__(self, config: dict | None = None):
        """모델 로드 및 초기화 로직이 들어갑니다."""
        pass

    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        """실제 AI 추론을 수행하는 핵심 메서드입니다."""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
