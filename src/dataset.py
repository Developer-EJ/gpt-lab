# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        # 마지막 target 토큰까지 필요하므로 context_length보다 토큰 1개가 더 있어야 합니다.
        self._length = (len(self.token_ids) - self.context_length - 1) // self.stride + 1

    def __len__(self) -> int:
        """DataLoader가 전체 학습 샘플 수를 알 수 있게 반환합니다."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 구간에서 input과 한 칸 뒤로 밀린 target을 만듭니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        # idx를 실제 token_ids의 시작 위치로 바꿉니다.
        start = idx * self.stride
        input_ids = self.token_ids[start : start + self.context_length]
        # 다음 토큰 예측을 학습하므로 target은 input보다 한 칸 앞선 구간입니다.
        target_ids = self.token_ids[start + 1 : start + self.context_length + 1]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """TODO: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""
    dataset = GPTDataset(token_ids, context_length, stride)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)