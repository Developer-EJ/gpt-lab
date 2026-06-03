# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _to_jsonable(value):
    """Path, device 같은 값을 JSON에 저장 가능한 값으로 바꿉니다."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    return str(value)


def _write_json(path: Path, data: dict) -> None:
    """dict를 보기 좋은 JSON 파일로 저장합니다."""

    path.write_text(
        json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict) -> None:
    """JSONL 파일에 한 줄짜리 기록을 추가합니다."""

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def _safe_perplexity(loss: float) -> float:
    """loss를 perplexity로 바꿉니다. 너무 큰 값은 inf로 둡니다."""

    if math.isnan(loss) or loss > 100:
        return float("inf")

    return math.exp(loss)


def _get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """첫 번째 optimizer parameter group의 learning rate를 반환합니다."""

    return optimizer.param_groups[0]["lr"]


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    """
    한 batch의 next-token prediction loss를 계산합니다.
    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    loss, logits = model(input_batch, targets=target_batch)

    return loss

    raise NotImplementedError("calc_loss_batch를 구현하세요.")


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""

    model.eval()

    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            total_loss += loss.item()

    model.train()

    return total_loss / num_batches

    raise NotImplementedError("calc_loss_loader를 구현하세요.")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
    config: dict | None = None,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    track_steps: list[int] | None = None,
    best_val_loss: float | None = None,
    best_step: int | None = None,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": _to_jsonable(config),
        "train_losses": train_losses or [],
        "val_losses": val_losses or [],
        "track_steps": track_steps or [],
        "best_val_loss": best_val_loss,
        "best_step": best_step,
    }

    torch.save(checkpoint, path)

    # raise NotImplementedError("save_checkpoint를 구현하세요.")


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    return epoch, global_step

    raise NotImplementedError("load_checkpoint를 구현하세요.")


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

    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            min_top_value = top_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_value,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

        if eos_id is not None and torch.all(next_token == eos_id):
            break

    return idx

    raise NotImplementedError("generate를 구현하세요.")


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> str:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""

    was_training = model.training
    model.eval()

    token_ids = tokenizer.encode(start_context, add_bos_eos=False)

    idx = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    safe_top_k = top_k
    if safe_top_k is not None:
        safe_top_k = min(safe_top_k, model.config["vocab_size"])

    generated = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=safe_top_k,
        eos_id=None,
    )

    generated_ids = generated[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special=True)

    print(generated_text)

    if was_training:
        model.train()

    return generated_text

    # raise NotImplementedError("generate_and_print_sample을 구현하세요.")


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
    run_dir: str | Path | None = None,
    run_name: str | None = None,
    config: dict | None = None,
) -> tuple[list[float], list[float], list[int]]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""

    model.to(device)
    model.train()

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"run_{timestamp}"
    else:
        run_dir = Path(run_dir)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.jsonl"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"

    model_config = getattr(model, "config", {})
    num_params = sum(param.numel() for param in model.parameters())

    train_dataset = getattr(train_loader, "dataset", None)
    val_dataset = getattr(val_loader, "dataset", None)
    train_tokens = len(getattr(train_dataset, "token_ids", []))
    val_tokens = len(getattr(val_dataset, "token_ids", []))

    run_config = {
        "run_name": run_name or run_dir.name,
        "device": str(device),
        "num_epochs": num_epochs,
        "eval_freq": eval_freq,
        "eval_iter": eval_iter,
        "ckpt_freq": ckpt_freq,
        "start_epoch": start_epoch,
        "start_global_step": global_step,
        "start_context": start_context,
        "model_config": model_config,
        "extra_config": config or {},
        "num_params": num_params,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "batch_size": getattr(train_loader, "batch_size", None),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(config_path, run_config)

    train_losses = []
    val_losses = []
    track_steps = []

    best_val_loss = float("inf")
    best_step = None
    start_time = time.time()
    initial_global_step = global_step

    tokens_per_step = (
        getattr(train_loader, "batch_size", None) or 0
    ) * model_config.get("context_length", 0)

    for epoch in range(start_epoch, num_epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                elapsed_sec = time.time() - start_time
                trained_steps = global_step - initial_global_step

                train_loss = calc_loss_loader(
                    data_loader=train_loader,
                    model=model,
                    device=device,
                    num_batches=eval_iter,
                )
                val_loss = calc_loss_loader(
                    data_loader=val_loader,
                    model=model,
                    device=device,
                    num_batches=eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_steps.append(global_step)

                metric = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": _safe_perplexity(train_loss),
                    "val_ppl": _safe_perplexity(val_loss),
                    "elapsed_sec": elapsed_sec,
                    "steps_per_sec": (
                        trained_steps / elapsed_sec if elapsed_sec > 0 else 0.0
                    ),
                    "tokens_per_sec": (
                        trained_steps * tokens_per_step / elapsed_sec
                        if elapsed_sec > 0
                        else 0.0
                    ),
                    "learning_rate": _get_learning_rate(optimizer),
                }
                _append_jsonl(metrics_path, metric)

                print(
                    f"epoch {epoch + 1:03d} | "
                    f"step {global_step:06d} | "
                    f"train loss {train_loss:.4f} | "
                    f"val loss {val_loss:.4f}"
                )

                generated_text = generate_and_print_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    start_context=start_context,
                    max_new_tokens=50,
                    context_size=model.config["context_length"],
                )
                _append_jsonl(
                    samples_path,
                    {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "prompt": start_context,
                        "generated_text": generated_text,
                        "temperature": 0.0,
                        "top_k": None,
                    },
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = global_step

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        path=str(checkpoint_dir / "best.pt"),
                        config=run_config,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        track_steps=track_steps,
                        best_val_loss=best_val_loss,
                        best_step=best_step,
                    )
                    print(f"best checkpoint saved: {checkpoint_dir / 'best.pt'}")

            if ckpt_freq is not None and global_step % ckpt_freq == 0:
                checkpoint_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=str(checkpoint_path),
                    config=run_config,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    track_steps=track_steps,
                    best_val_loss=best_val_loss,
                    best_step=best_step,
                )

                print(f"checkpoint saved: {checkpoint_path}")

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=str(checkpoint_dir / "last.pt"),
                    config=run_config,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    track_steps=track_steps,
                    best_val_loss=best_val_loss,
                    best_step=best_step,
                )

    total_train_time_sec = time.time() - start_time

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs - 1,
        global_step=global_step,
        path=str(checkpoint_dir / "last.pt"),
        config=run_config,
        train_losses=train_losses,
        val_losses=val_losses,
        track_steps=track_steps,
        best_val_loss=best_val_loss,
        best_step=best_step,
    )

    summary = {
        "run_name": run_config["run_name"],
        "total_train_time_sec": total_train_time_sec,
        "final_step": global_step,
        "best_step": best_step,
        "best_val_loss": best_val_loss if best_step is not None else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "num_eval_points": len(track_steps),
        "metrics_path": metrics_path,
        "samples_path": samples_path,
        "last_checkpoint": checkpoint_dir / "last.pt",
        "best_checkpoint": (
            checkpoint_dir / "best.pt" if best_step is not None else None
        ),
    }
    _write_json(summary_path, summary)

    return train_losses, val_losses, track_steps

    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    steps: list[int] | None = None,
) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""

    x_values = steps if steps is not None else list(range(1, len(train_losses) + 1))

    plt.plot(x_values, train_losses, label="Train")
    if val_losses is not None:
        plt.plot(x_values, val_losses, label="Val")
    plt.xlabel("Step" if steps is not None else "Eval")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
