# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # 사전 학습된 GPT backbone은 문장 표현을 만드는 역할로 사용합니다.
        self.gpt = gpt_model
        # 감성 분류에서는 보통 긍정/부정 2개 class를 예측합니다.
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        # 분류 head 앞에 dropout을 두어 미세 조정 중 과적합을 줄입니다.
        self.dropout = nn.Dropout(drop_rate)
        # LM head는 그대로 두고, hidden state 위에 별도 분류 head를 붙입니다.
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # GPTModel.forward는 LM head까지 지나가므로, 여기서는 hidden state 흐름만 직접 사용합니다.
        x = self.gpt.embedding(input_ids)
        x = self.gpt.blocks(x)
        x = self.gpt.final_norm(x)

        # 문장 전체를 대표할 벡터로 마지막 토큰 위치의 hidden state를 사용합니다.
        last_token_hidden = x[:, -1, :]
        # 별도 classifier head가 hidden state를 감성 class logits로 바꿉니다.
        logits = self.classifier(self.dropout(last_token_hidden))

        if labels is None:
            return logits

        # labels가 주어지면 분류용 cross entropy loss를 함께 반환합니다.
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    # 학습 epoch이므로 dropout 등을 켠 train 모드로 전환합니다.
    model.train()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        # batch tensor를 모델과 같은 device로 이동합니다.
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 분류 loss로 역전파를 수행하고 optimizer가 파라미터를 갱신합니다.
        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        # 평균 loss를 샘플 기준으로 계산하기 위해 batch 크기만큼 가중해 누적합니다.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        # 가장 큰 logit의 class를 예측값으로 보고 accuracy를 계산합니다.
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")

    # 한 epoch 전체의 평균 loss와 정답 비율을 반환합니다.
    return total_loss / total_examples, correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    total_loss = 0.0
    correct = 0
    total_examples = 0
    # 평가가 끝난 뒤 원래 학습/평가 상태로 되돌리기 위해 현재 mode를 기억합니다.
    was_training = model.training

    # 평가 중에는 dropout을 끄고 gradient를 만들지 않습니다.
    model.eval()
    with torch.no_grad():
        for input_ids, labels in data_loader:
            # batch tensor를 모델과 같은 device로 이동합니다.
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels=labels)
            batch_size = labels.size(0)
            # 평균 loss를 샘플 기준으로 계산하기 위해 batch 크기만큼 가중해 누적합니다.
            total_loss += loss.item() * batch_size
            # 가장 큰 logit을 예측 class로 사용해 정답 개수를 셉니다.
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total_examples += batch_size

    # 호출 전 train 모드였다면 평가 후 다시 train 모드로 복구합니다.
    if was_training:
        model.train()

    if total_examples == 0:
        return float("nan"), float("nan")

    # 전체 평가 데이터 기준 평균 loss와 accuracy를 반환합니다.
    return total_loss / total_examples, correct / total_examples
