# -*- coding: utf-8 -*-
"""
04-04-04 word2vec속도 개선 — Embedding 계층과네거티브 샘플링

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-04 word2vec속도 개선 — Embedding 계층과네거티브 샘플링.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-04-A Embedding 계층 — "곱하지 말고, 그냥 꺼내오자"
import numpy as np

class Embedding:
    """임베딩 계층: 단어 ID → 해당 행의 벡터
    비유: '도서관 직접 찾기' — 모든 책장을 훑지 않고 번호로 바로 꺼내기"""

    def __init__(self, W):
        self.W = W                           # 가중치(=임베딩) 행렬
        self.idx = None                     # 역전파용 인덱스 캐싱

    def forward(self, idx):
        """순전파: 단어 ID로 행 꺼내기 (행렬곱 없음!)"""
        self.idx = idx                       # 어떤 행을 꺼냈는지 기록
        return self.W[idx]                   # 인덱싱 한 줄로 끝!

    def backward(self, dout):
        """역전파: 해당 행에만 기울기 누적"""
        dW = np.zeros_like(self.W)          # 전체를 0으로
        np.add.at(dW, self.idx, dout)       # 꺼낸 행에만 기울기 더하기
        return dW

# 테스트: 7개 단어 × 3차원 임베딩
W = np.arange(21).reshape(7, 3)
emb = Embedding(W)

print("전체 가중치 W:")
print(W)
print("\n'say'(ID=1)의 임베딩:", emb.forward(1))
print("'goodbye'(ID=2)의 임베딩:", emb.forward(2))


# %% [Block 2] 04-04-04-B 네거티브 샘플링 — "100만 개 중 정답 찾기"를 "O/X 6번"으로
import numpy as np

def sigmoid(x):
    """시그모이드: 입력을 0~1 사이로 압축하는 S자 함수"""
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y, t):
    """이진 교차 엔트로피: O/X 문제의 채점 함수"""
    eps = 1e-7
    return -(t * np.log(y + eps) + (1 - t) * np.log(1 - y + eps))

# 긍정 샘플: "이 쌍은 진짜다" → 정답=1, score가 클수록 좋음
pos_score = 2.5
pos_y = sigmoid(pos_score)
print("긍정 샘플 확률:", round(pos_y, 4))
print("긍정 샘플 손실:", round(binary_cross_entropy(pos_y, 1), 4))

# 부정 샘플: "이 쌍은 가짜다" → 정답=0, score가 작을수록 좋음
neg_score = -1.8
neg_y = sigmoid(neg_score)
print("부정 샘플 확률:", round(neg_y, 4))
print("부정 샘플 손실:", round(binary_cross_entropy(neg_y, 0), 4))
