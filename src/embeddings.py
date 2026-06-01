# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # 각 토큰별 벡터 값 생성(초기에는 랜덤값)
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # 절대 위치 임베딩
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len = x.shape
        # 1. 각 토큰 ID를 벡터로 변환(이미 만들어진 사전에서 값을 조회)
        token_emb = self.token_embedding(x)
        # 2. 각 위치를 벡터로 변환
        positions = torch.arange(seq_len, device=x.device)
        # 3. 토큰 의미 + 위치 정보를 더함
        pos_emb = self.position_embedding(positions)
        # 4. 과적합 방지를 위해 일부 값을 0으로 바꾸고, 최종 실수 벡터 return
        return self.dropout(token_emb + pos_emb)
