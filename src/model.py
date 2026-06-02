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
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # 각 token의 hidden vector 내부에서만 평균/분산을 구해 batch와 sequence 구조는 유지합니다.
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normed + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.net(x)


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
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        drop_rate = config.get("drop_rate", 0.1)
        qkv_bias = config.get("qkv_bias", False)

        # config 값만 바꾸면 작은 smoke 모델부터 더 큰 실험 모델까지 같은 코드로 만들 수 있습니다.
        self.embedding = InputEmbedding(vocab_size, emb_dim, context_length, drop_rate=drop_rate)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim,
                    n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """토큰 ID를 GPT backbone hidden state로 변환합니다."""
        x = self.embedding(idx)
        # 여러 TransformerBlock을 통과한 마지막 hidden state가 LM/분류 head의 공통 입력입니다.
        for block in self.blocks:
            x = block(x, causal_mask=True)
        return self.final_norm(x)

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
        hidden = self.forward_hidden(idx)
        logits = self.lm_head(hidden)

        if targets is None:
            return logits

        # F.cross_entropy는 (N, C) logits와 (N,) target을 기대하므로 batch/sequence 축을 펼칩니다.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        # context window보다 긴 prefix가 들어오면 마지막 context_size 토큰만 모델에 넣습니다.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        # greedy 생성은 마지막 위치의 vocab logits에서 가장 큰 token ID를 그대로 선택합니다.
        idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
