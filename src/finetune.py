# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def _read_nsmc_tsv(path: str | Path) -> list[dict]:
        # NSMC TSV에서 리뷰 본문(document)과 정답(label)만 표준 dict 형태로 추출합니다.
        rows = []
        with open(Path(path), "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("document", "")
                label = row.get("label", "")
                # 빈 리뷰는 tokenizer와 모델 입력으로 쓰기 어렵기 때문에 제외합니다.
                if text is None or text.strip() == "":
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    train_rows = _read_nsmc_tsv(train_tsv_path)
    # seed를 고정한 shuffle로 매번 같은 train/validation split을 만들 수 있게 합니다.
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    # train 파일 일부를 validation으로 떼어내고, 나머지를 train으로 사용합니다.
    val_size = int(len(train_rows) * val_ratio)
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    # test 파일이 있으면 같은 형식으로 읽고, 없으면 빈 리스트를 반환합니다.
    test_data = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        # 전처리 결과를 재사용할 수 있도록 선택적으로 JSON 파일로 저장합니다.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("train.json", train_data),
            ("val.json", val_data),
            ("test.json", test_data),
        ):
            with open(output_path / name, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        # make_sentiment_dataset이 만든 {"text", "label"} 형식의 데이터를 보관합니다.
        self.data = data
        # __getitem__에서 리뷰 문자열을 token id로 바꿀 tokenizer를 보관합니다.
        self.tokenizer = tokenizer
        # 모든 샘플을 같은 길이로 맞추기 위한 최대 token 길이입니다.
        self.max_length = max_length
        # pad_id가 직접 주어지지 않으면 tokenizer의 padding ID를 사용합니다.
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        # DataLoader가 전체 샘플 개수를 알 수 있도록 원본 데이터 길이를 반환합니다.
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        # idx번째 리뷰와 정답 label을 꺼냅니다.
        item = self.data[idx]
        # 리뷰 문자열을 모델 입력으로 사용할 token id 리스트로 변환합니다.
        token_ids = self.tokenizer.encode(item["text"])

        # 너무 긴 리뷰는 max_length까지만 사용합니다.
        token_ids = token_ids[: self.max_length]
        if len(token_ids) < self.max_length:
            # 짧은 리뷰는 batch로 묶을 수 있도록 pad_id로 길이를 맞춥니다.
            token_ids += [self.pad_id] * (self.max_length - len(token_ids))

        # embedding layer가 받을 수 있도록 token id를 long tensor로 반환합니다.
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        return input_ids, int(item["label"])


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("train_epoch_sentiment를 구현하세요.")


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("evaluate_sentiment를 구현하세요.")
