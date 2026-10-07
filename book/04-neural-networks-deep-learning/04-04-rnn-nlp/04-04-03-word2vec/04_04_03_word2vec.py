# -*- coding: utf-8 -*-
"""
04-04-03 word2vec— 추론 기반 단어임베딩

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-03 word2vec— 추론 기반 단어임베딩.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-03-A CBOW 모델 — 주변 단어로 빈칸 맞추기
import numpy as np

# --- 준비 ---
V, H = 7, 3   # V=어휘 크기 7, H=임베딩 차원 3
np.random.seed(42)
W_in  = np.random.randn(V, H) * 0.01   # 입력 가중치 (7×3)
W_out = np.random.randn(H, V) * 0.01   # 출력 가중치 (3×7)

# --- 1단계: 맥락 단어를 원-핫 벡터로 ---
c_you     = np.array([1,0,0,0,0,0,0])   # "you" = ID 0
c_goodbye = np.array([0,0,1,0,0,0,0])   # "goodbye" = ID 2

# --- 2단계: 원-핫 × W_in = W_in의 해당 행을 꺼냄 ---
h_you     = c_you @ W_in                  # (7,)@(7,3)=(3,)
h_goodbye = c_goodbye @ W_in
print("h_you:    ", h_you.round(4))
print("h_goodbye:", h_goodbye.round(4))

# --- 3단계: 두 맥락의 평균 → 은닉층 ---
h = (h_you + h_goodbye) / 2               # 여러 단서를 종합
print("은닉층 h: ", h.round(4))

# --- 4단계: 은닉층 × W_out → 점수 → 소프트맥스 ---
def softmax(x):
    """소프트맥스: 점수를 확률분포로 변환 (총합=1)"""
    x = x - np.max(x)                      # 오버플로 방지
    return np.exp(x) / np.sum(np.exp(x))

score = h @ W_out                         # (3,)@(3,7)=(7,)
prob = softmax(score)
print("각 단어 확률:", prob.round(3))


# %% [Block 2] 04-04-03-C Skip-gram과 "왕 - 남자 + 여자 = 여왕" — 학습이 완료된 임베딩이 있다고 가정
# 학습이 완료된 임베딩이 있다고 가정
# vec("왕") - vec("남자") + vec("여자") ≈ vec("여왕")
#
# 해석:
# 1. "왕" - "남자" = "왕족/통치자"라는 방향 성분
# 2. 여기에 "여자"를 더하면 = "여성 + 왕족/통치자" = "여왕"
#
# 신경망이 여러 의미 축(성별, 직위 등)을
# 벡터의 여러 차원에 나누어 인코딩했기 때문!
