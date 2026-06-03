# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    # 모델과 같은 장치에서 계산되도록 입력과 정답 batch를 이동합니다.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # GPTModel은 targets가 있으면 내부에서 cross entropy loss를 함께 반환합니다.
    loss, _ = model(input_batch, targets=target_batch)
    # 학습 루프에서는 logits보다 loss가 필요하므로 loss만 돌려줍니다.
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0.0
    num_seen_batches = 0
    # 호출 전 학습/평가 상태를 기억해 두었다가 마지막에 되돌립니다.
    was_training = model.training

    # loader loss는 평가 목적이므로 dropout 등을 끄고 gradient도 만들지 않습니다.
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            # num_batches가 지정되면 전체 loader 대신 앞쪽 일부 batch만 평가합니다.
            if num_batches is not None and batch_idx >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_seen_batches += 1

    # 이 함수를 호출하기 전 train 모드였다면 다시 train 모드로 복구합니다.
    if was_training:
        model.train()

    if num_seen_batches == 0:
        return float("nan")

    # batch별 loss의 산술 평균을 반환합니다.
    return total_loss / num_seen_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # 학습을 이어서 재개할 수 있도록 모델, 옵티마이저, 진행 위치를 함께 저장합니다.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # 저장된 tensor를 현재 실행 device에 맞춰 불러옵니다.
    checkpoint = torch.load(path, map_location=device)
    # 모델 가중치를 checkpoint 상태로 되돌립니다.
    model.load_state_dict(checkpoint["model_state_dict"])

    # 추론만 할 때는 optimizer가 없을 수 있으므로, 주어진 경우에만 복원합니다.
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 학습 루프가 이어서 시작할 epoch과 step을 알 수 있게 반환합니다.
    return checkpoint["epoch"], checkpoint["global_step"]


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    raise NotImplementedError("generate를 구현하세요.")


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
