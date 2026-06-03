# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

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
    import csv
    import random

    random.seed(seed)

    def read_tsv(path):
        samples = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("document", "").strip()
                label = row.get("label", "").strip()
                if text and label in ("0", "1"):
                    samples.append({"text": text, "label": int(label)})
        return samples

    # train TSV 읽고 train/val 분리
    all_train = read_tsv(train_tsv_path)
    random.shuffle(all_train)
    val_size = int(len(all_train) * val_ratio)
    val_data = all_train[:val_size]
    train_data = all_train[val_size:]

    # test TSV가 있으면 읽고, 없으면 빈 리스트
    test_data = read_tsv(test_tsv_path) if test_tsv_path is not None else []

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
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # 텍스트를 토큰 ID로 변환 후 max_length에 맞게 자르거나 pad_id로 패딩
        item = self.data[idx]
        token_ids = self.tokenizer.encode(item["text"])[: self.max_length]
        pad_len = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * pad_len
        return torch.tensor(token_ids, dtype=torch.long), item["label"]


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
        # GPT hidden state를 받아 num_labels 클래스로 분류하는 헤드
        d_model = gpt_model.config.get("d_model") or gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # GPT backbone으로 lm_head 이전 hidden state 추출 후 마지막 토큰 벡터를 문장 대표로 사용
        hidden = self.gpt(input_ids, return_hidden=True)  # (B, T, d_model)
        cls_vec = hidden[:, -1, :]  # 마지막 토큰 벡터를 문장 대표로 사용
        logits = self.classifier(self.dropout(cls_vec))  # (B, num_labels)

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    # 1 epoch 동안 배치별 loss.backward() + optimizer.step() 반복 후 평균 loss/accuracy 반환
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    # gradient 불필요하므로 no_grad, 전체 데이터로 평균 loss/accuracy 계산
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels)
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), correct / total
