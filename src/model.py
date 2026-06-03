# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape)) # 초기값 1
        self.beta = nn.Parameter(torch.zeros(normalized_shape)) # 초기값 0
        self.eps = eps # 0으로 나누기 방지 (1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # 각 토큰의 hidden vector(d_model) 안에서만 평균을 구합니다.
        # batch나 sequence 길이가 아니라 마지막 차원만 정규화 대상입니다.
        x_mean = x.mean(dim=-1, keepdim=True)
        # LayerNorm은 현재 벡터 자체를 정규화하므로 표본 보정 없이 분산을 계산합니다.
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        # eps는 분산이 0에 가까울 때 나눗셈이 불안정해지는 것을 막습니다.
        x_norm = (x - x_mean) / (x_var + self.eps).sqrt()
        # 정규화한 값에 학습 가능한 scale(gamma)과 shift(beta)를 적용합니다.
        return self.gamma * x_norm + self.beta

class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        # GELU는 입력을 0/1로 딱 자르지 않고, 값의 크기에 따라 부드럽게 통과시킵니다.
        # 아래 식은 GPT 계열에서 자주 쓰는 tanh 기반 GELU 근사식이며 입력 shape는 그대로 유지됩니다.
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        # 토큰별 hidden vector를 더 넓은 차원으로 확장한 뒤 다시 d_model로 되돌립니다.
        self.layers = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            # GELU는 확장된 표현에 비선형성을 추가해 더 복잡한 패턴을 표현하게 합니다.
            GELU(),
            nn.Linear(mult * d_model, d_model),
            # Dropout은 학습 중 일부 값을 끄며 과적합을 줄이는 역할을 합니다.
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        # 입력과 출력 shape는 같고, 내부에서만 차원이 잠시 확장됩니다.
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        # 첫 번째 sublayer: 정규화한 입력을 causal self-attention에 통과시킵니다.
        self.att = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        # 두 번째 sublayer: attention 결과를 토큰별 MLP로 한 번 더 변환합니다.
        self.ff = FeedForward(d_model, dropout=drop_rate)
        # GPT block은 sublayer 앞에서 LayerNorm을 적용하는 pre-LN 구조입니다.
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # residual에 더하기 전 sublayer 출력에 dropout을 적용합니다.
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # Attention sublayer: norm -> attention -> dropout -> residual add
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, causal_mask=causal_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # FeedForward sublayer: norm -> FFN -> dropout -> residual add
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        # token id와 position 정보를 Transformer가 처리할 수 있는 vector로 바꿉니다.
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )
        # 같은 구조의 TransformerBlock을 n_layers개 쌓아 문맥 정보를 처리합니다.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=config["emb_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                )
                for _ in range(config["n_layers"])
            ]
        )
        # 모든 block을 지난 뒤 마지막으로 hidden vector의 스케일을 정리합니다.
        self.final_norm = LayerNorm(config["emb_dim"])
        # 각 위치의 hidden vector를 vocab_size 차원의 다음 토큰 점수로 바꿉니다.
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        # token id 입력을 embedding vector 시퀀스로 변환합니다.
        x = self.embedding(idx)
        # TransformerBlock들이 이전 token 문맥을 반영한 표현으로 갱신합니다.
        x = self.blocks(x)
        # 출력 직전 LayerNorm으로 hidden state를 안정화합니다.
        x = self.final_norm(x)
        # 각 위치마다 vocabulary 전체에 대한 다음 token 점수(logits)를 만듭니다.
        logits = self.lm_head(x)

        # targets가 있으면 학습 모드처럼 정답 token과 비교해 loss를 함께 계산합니다.
        if targets is not None:
            # cross entropy는 (N, C) logits와 (N,) target 형태를 기대하므로 토큰 위치를 펼칩니다.
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
            )
            return loss, logits

        # 추론이나 생성에서는 loss 없이 logits만 사용합니다.
        return logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    # 새 토큰을 하나씩 예측해 기존 token sequence 뒤에 붙입니다.
    for _ in range(max_new_tokens):
        # 모델의 최대 context 길이를 넘지 않도록 마지막 context_size개 토큰만 사용합니다.
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # 다음 토큰은 현재 sequence의 마지막 위치 logits에서 고릅니다.
        logits = logits[:, -1, :]
        # greedy decoding: 확률이 가장 높은 token id를 선택합니다.
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # 선택한 token을 sequence 뒤에 이어 붙입니다.
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
