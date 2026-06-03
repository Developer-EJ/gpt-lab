# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        # nn.Linear가 하는 일 : 행렬 곱
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        # 입력에서 batch 크기와 sequence 길이를 가져옵니다.
        b, t, _ = x.shape

        # 입력을 query, key, value 벡터로 변환합니다.
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 각 head가 따로 attention을 계산하도록 차원을 나눕니다.
        queries = queries.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # query와 key의 유사도로 attention score를 만듭니다.
        result = queries @ keys.transpose(-2, -1) / self.head_dim ** 0.5

        # causal mask로 미래 token 위치를 가립니다.
        if causal_mask:
            mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
            result = result.masked_fill(mask, float('-inf'))

        # score를 확률로 바꾼 뒤 value와 곱해 새 token 표현을 만듭니다.
        result_soft = self.drop_out(torch.softmax(result, dim=-1))

        new_vector = result_soft @ values

        # 나누었던 head를 다시 합쳐 d_model 차원으로 되돌립니다.
        new_vector = new_vector.transpose(1, 2).contiguous()
        new_vector = new_vector.view(b, t, self.d_model)

        # 마지막 선형 변환으로 attention 출력을 정리합니다.
        out_proj = self.out_proj(new_vector)

        if return_attention_weights:
            return out_proj, result_soft
        else:
            return out_proj
