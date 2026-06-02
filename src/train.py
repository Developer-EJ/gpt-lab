# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
    # 학습/평가 모두 같은 helper를 쓰기 위해 batch를 여기서 한 번에 device로 옮깁니다.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    if len(data_loader) == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    losses = []
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    # validation loss 계산은 gradient가 필요 없으므로 메모리와 시간을 아끼기 위해 no_grad를 사용합니다.
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            losses.append(loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # optimizer와 진행 위치를 함께 저장해야 Colab 런타임 재시작 후 같은 지점에서 이어서 학습할 수 있습니다.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # 추론 용도로 불러올 때는 optimizer가 없을 수 있으므로 선택적으로 복원합니다.
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


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
    for _ in range(max_new_tokens):
        # 학습 때 본 최대 길이를 넘지 않도록 최근 context_size 토큰만 조건으로 사용합니다.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            # top-k 밖의 token은 -inf로 막아 softmax 이후 선택될 확률을 0으로 만듭니다.
            top_k = min(top_k, logits.size(-1))
            top_values, _ = torch.topk(logits, top_k)
            min_top_value = top_values[:, [-1]]
            logits = torch.where(logits < min_top_value, torch.full_like(logits, float("-inf")), logits)

        if temperature == 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # temperature가 낮을수록 높은 logit에 더 몰리고, 높을수록 더 다양한 token을 샘플링합니다.
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        if eos_id is not None and torch.all(idx_next == eos_id):
            break

    return idx


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
    was_training = model.training
    model.eval()
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    out = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None,
    )
    print(tokenizer.decode(out[0].tolist()))
    if was_training:
        model.train()


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
    model.to(device)
    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_losses = []

        for input_batch, target_batch in train_loader:
            # 표준 PyTorch 학습 순서: gradient 초기화 -> forward/loss -> backward -> optimizer step.
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())

            if eval_freq and global_step % eval_freq == 0:
                calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

            if ckpt_freq and global_step % ckpt_freq == 0:
                # 긴 학습 중간에 주기적으로 저장해 런타임 종료나 실험 중단에 대비합니다.
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

        train_losses.append(sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan"))
        if start_context and tokenizer is not None:
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
                context_size=model.config["context_length"],
            )

    return train_losses


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
