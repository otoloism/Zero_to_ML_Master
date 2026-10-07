# -*- coding: utf-8 -*-
"""
04-04-08 어텐션—Transformer로 가는 다리

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-08 어텐션—Transformer로 가는 다리.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-08-A 어텐션 3단계를 NumPy로 직접 구현
import numpy as np

def softmax(x):
    """안전한 소프트맥스 (오버플로 방지)"""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    """Scaled Dot-Product Attention
    비유: '도서관 사서가 질문에 맞는 책을 골라 요약해 주는 과정'"""
    d_k = K.shape[-1]

    # ① 내적: 질문(Q)과 각 책 제목(K)의 관련도
    scores = Q @ K.T                     # (N, N)

    # ② 스케일링 + 소프트맥스: 확률로 변환
    alpha = softmax(scores / np.sqrt(d_k)) # √d로 나눠 안정화

    # ③ 가중합: 관련 비율대로 책 내용(V) 요약
    context = alpha @ V                  # (N, d_v)

    return context, alpha

# 예시: 3개 단어, 임베딩 차원 4
np.random.seed(42)
seq_len, d = 3, 4
Q = np.random.randn(seq_len, d)
K = np.random.randn(seq_len, d)
V = np.random.randn(seq_len, d)

ctx, alpha = attention(Q, K, V)

print("어텐션 가중치 α (어디에 주목했나?):")
print(alpha.round(3))
print("\n각 행의 합 =", alpha.sum(axis=1).round(3), "← 항상 1 (확률이므로)")
