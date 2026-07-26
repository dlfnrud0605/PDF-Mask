import math
import torch
from collections import defaultdict
from torch.utils.data import BatchSampler
from mmengine.hooks import Hook
from mmengine.registry import HOOKS, FUNCTIONS
from mmocr.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SizeAwareBatchSampler(BatchSampler):
    """데이터셋의 실제 크기 분포 기반 3버킷 배치 샘플러.

    버킷:
      wide    — w > h이고 w/h > 1.5  (16:9 계열, ~83%)
      portrait — h > w               (세로 이미지, ~15%)
      other   — 나머지               (정방형·4:3 계열, ~3%)

    같은 버킷끼리만 배치를 구성해 dynamic_pad_collate의 패딩 낭비를 최소화함.
    """

    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # other 버킷(1280×1280 등)은 절반 배치로 OOM 방지
        self._bucket_bs = {0: batch_size, 1: batch_size, 2: max(1, batch_size // 2)}
        # __len__ 정확도를 위해 초기화 시 버킷별 카운트 미리 계산
        counts = defaultdict(int)
        for idx in range(len(sampler.dataset)):
            info = sampler.dataset.get_data_info(idx)
            b = self._bucket(info.get('height', 1), info.get('width', 1))
            counts[b] += 1
        self._bucket_counts = counts

    @staticmethod
    def _bucket(h, w):
        if h > w:
            return 0  # portrait
        elif w / max(h, 1) > 1.5:
            return 1  # wide
        else:
            return 2  # other (square / 4:3)

    def __iter__(self):
        buckets = defaultdict(list)
        for idx in self.sampler:
            info = self.sampler.dataset.get_data_info(idx)
            h = info.get('height', 1)
            w = info.get('width', 1)
            b = self._bucket(h, w)
            buckets[b].append(idx)
            if len(buckets[b]) == self._bucket_bs[b]:
                yield list(buckets[b])
                buckets[b] = []

        if not self.drop_last:
            leftover = [idx for idxs in buckets.values() for idx in idxs]
            for i in range(0, len(leftover), self.batch_size):
                yield leftover[i:i + self.batch_size]

    def __len__(self):
        total = 0
        for b, count in self._bucket_counts.items():
            bs = self._bucket_bs[b]
            if self.drop_last:
                total += count // bs
            else:
                total += (count + bs - 1) // bs
        return total


@FUNCTIONS.register_module()
def dynamic_pad_collate(data_batch):
    """배치 내 최대 크기(32 배수)로 동적 패딩 후 collate.

    AspectRatioBatchSampler와 함께 사용. 비율이 비슷한 이미지끼리 묶인
    배치 내에서 최소한의 패딩만 적용해 불필요한 연산을 줄임.
    img_shape도 패딩된 크기로 업데이트해 DBModuleLoss의 gt_shrinks 크기를 일치시킴.
    """
    inputs = [sample['inputs'] for sample in data_batch]
    data_samples = [sample['data_samples'] for sample in data_batch]

    # 배치 내 최대 H, W를 32 배수로 올림
    max_h = math.ceil(max(img.shape[1] for img in inputs) / 32) * 32
    max_w = math.ceil(max(img.shape[2] for img in inputs) / 32) * 32

    padded_inputs = []
    for img in inputs:
        c, h, w = img.shape
        padded = torch.zeros(c, max_h, max_w, dtype=img.dtype)
        padded[:, :h, :w] = img
        padded_inputs.append(padded)

    # img_shape을 패딩된 크기로 업데이트 (DBModuleLoss gt 생성 기준)
    for ds in data_samples:
        ds.set_metainfo({'img_shape': (max_h, max_w)})

    return {'inputs': padded_inputs, 'data_samples': data_samples}


@HOOKS.register_module()
class CheckpointKeyGuard(Hook):
    """
    학습 시작 전 load_from 가중치가 모델에 정상 로드됐는지 검사.
    critical_prefix에 해당하는 레이어 키가 10% 이상 누락되면 즉시 오류 발생.
    mmengine의 strict=False 묵인 동작을 보완함.
    """

    def __init__(self, checkpoint_path: str, critical_prefix: str = 'backbone'):
        self.checkpoint_path = checkpoint_path
        self.critical_prefix = critical_prefix

    def before_train(self, runner) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # checkpoint 키에서 'model.' 접두사 제거 (mmocr 저장 형식 대응)
        ckpt_keys = set()
        for k in state_dict.keys():
            clean = k[len('model.'):] if k.startswith('model.') else k
            ckpt_keys.add(clean)

        # 모델의 critical_prefix 해당 키 목록
        model_keys = {
            k for k in runner.model.state_dict().keys()
            if k.startswith(self.critical_prefix)
        }

        missing = model_keys - ckpt_keys
        total = len(model_keys)
        loaded = total - len(missing)

        if total == 0:
            raise RuntimeError(
                f"[CheckpointKeyGuard] 모델에 '{self.critical_prefix}' 레이어가 없습니다. "
                f"critical_prefix 설정을 확인하세요."
            )

        ratio = len(missing) / total
        runner.logger.info(
            f"[CheckpointKeyGuard] '{self.critical_prefix}' 가중치 로드: "
            f"{loaded}/{total}개 매칭 (누락 {len(missing)}개, {ratio:.1%})"
        )

        if len(missing) > 0:
            missing_examples = list(missing)[:5]
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"[CheckpointKeyGuard] 오류: '{self.critical_prefix}' 가중치 {len(missing)}/{total}개({ratio:.1%}) 누락!\n"
                f"checkpoint와 모델 아키텍처가 불일치합니다.\n"
                f"누락 키 예시: {missing_examples}\n"
                f"checkpoint 경로: {self.checkpoint_path}\n"
                f"※ 키 이름 불일치가 아닌 실제 로드 여부가 확실하면 이 훅을 제거하고 진행하세요.\n"
                f"{'='*60}"
            )
