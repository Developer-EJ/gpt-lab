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
    # 배치를 device로 옮긴 뒤 모델에 넣어 loss 반환 (GPTModel.forward가 targets 있으면 loss 반환)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    # num_batches만큼(없으면 전체) 배치를 순회하며 평균 loss 계산, gradient 불필요하므로 no_grad
    total_loss = 0.0
    num_batches = num_batches or len(data_loader)
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    # 학습 재개에 필요한 모델/옵티마이저 가중치와 진행 상태를 파일로 저장
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    # 저장된 checkpoint를 불러와 모델/옵티마이저 상태 복원 후 epoch, global_step 반환
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
    # 매 스텝마다 마지막 토큰의 logits에 temperature/top-k 적용 후 샘플링으로 다음 토큰 생성
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # top-k: 상위 k개 제외한 나머지를 -inf로 마스킹
        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            logits[logits < top_values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

        if eos_id is not None and (next_token == eos_id).all():
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
    # 시작 문자열을 토큰 ID로 변환 후 생성, 결과를 디코딩해서 출력
    model.eval()
    idx = torch.tensor(tokenizer.encode(start_context), dtype=torch.long).unsqueeze(0).to(device)
    result = generate(model, idx, max_new_tokens, context_size, temperature, top_k)
    print(tokenizer.decode(result[0].tolist()))
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
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()       # gradient 계산
            optimizer.step()      # W 업데이트
            epoch_loss += loss.item()
            global_step += 1

            # eval_freq 스텝마다 train/val loss 출력
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(f"epoch {epoch+1} | step {global_step} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        # epoch 단위 평균 loss 기록
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(calc_loss_loader(val_loader, model, device, eval_iter))

        # 샘플 텍스트 생성으로 학습 진행 확인
        generate_and_print_sample(model, tokenizer, device, start_context)

        # ckpt_freq 에폭마다 체크포인트 저장
        if ckpt_freq is not None and (epoch + 1) % ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, f"checkpoint_epoch{epoch+1}.pt")

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
