import yaml
import importlib
from pathlib import Path

class ModelFactory:
    """
    설정 파일(pipeline.yaml)을 읽어 필요한 AI 모델 부품을 동적으로 생성해주는 공장(Factory)입니다.
    """
    _cache = {}

    @staticmethod
    def get_processor(module_type: str):
        """
        module_type: 'layout_detector', 'text_detector', 'recognizer', 'keyword_extractor' 중 하나
        """
        if module_type in ModelFactory._cache:
            return ModelFactory._cache[module_type]

        # 1. 설정 파일 읽기
        config_path = Path("configs/pipeline.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        # 2. 어떤 모델을 써야 하는지 메뉴판(pipeline)에서 확인
        model_name = config['pipeline'].get(module_type)
        if not model_name:
            raise ValueError(f"[{module_type}]에 해당하는 모델이 설정되어 있지 않습니다.")
            
        # 3. 모델 코드가 도대체 어디 있는지 설정 파일의 레지스트리(registry)에서 찾기
        registry = config.get('registry', {})
        class_path = registry.get(model_name)
        if not class_path:
            raise ValueError(f"알 수 없는 모델입니다. pipeline.yaml의 registry에 {model_name}의 경로를 추가해주세요.")
            
        # 4. 동적으로 파이썬 코드(.py) 가져오기 (import)
        module_path, class_name = class_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except ImportError:
            raise ImportError(f"{model_name} 모델의 코드 파일을 찾을 수 없습니다. ({module_path}.py 코드를 먼저 만들어주세요.)")
        
        # 5. 해당 모델 전용 세부 설정(가중치 경로, GPU 여부)만 잘라서 넘겨주며 부품 생성!
        # 예: "YOLOv8" -> 소문자로 바꿔서 "yolov8" 항목을 찾음
        model_config = config['models'].get(model_name.lower().replace(" ", "_"), {})
        
        instance = model_class(model_config)
        ModelFactory._cache[module_type] = instance
        return instance
