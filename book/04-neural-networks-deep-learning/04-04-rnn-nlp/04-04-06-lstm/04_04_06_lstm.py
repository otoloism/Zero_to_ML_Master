# -*- coding: utf-8 -*-
"""
04-04-06 게이트가 추가된RNN—LSTM과 기울기 문제 해결

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-06 게이트가 추가된RNN—LSTM과 기울기 문제 해결.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-06-A 게이트의 정체: 시그모이드 = 0~1 사이의 "수도꼭지"
import numpy as np

def sigmoid(x):
    """시그모이드: 0~1 사이 값으로 압축 = '수도꼭지 비율'"""
    return 1 / (1 + np.exp(-x))

# 게이트 입력 [-2, 0, 2] → 시그모이드 → [닫힘, 반쯤, 열림]
gate = sigmoid(np.array([-2, 0, 2]))
info = np.array([10, 10, 10])           # 통과시킬 정보

print("게이트 비율:", gate.round(3))     # [0.119, 0.5, 0.881]
print("통과 후 정보:", (gate * info).round(3))  # [1.19, 5.0, 8.81]
print("→ 게이트≈0: 정보 차단 / 게이트≈1: 정보 통과!")


# %% [Block 2] 04-04-06-B LSTM 한 단계 직접 계산하기
def lstm_step(x, h_prev, c_prev, Wx, Wh, b):
    """LSTM 1단계 순전파
    비유: '냉장고 관리 1회전' — 청소+장보기+꺼내기를 한 번에"""
    H = h_prev.shape[0]
    # 4개 게이트의 입력을 한 번에 계산 (효율적!)
    A = x @ Wx + h_prev @ Wh + b          # (4H,) 크기

    # 4등분하여 각 게이트에 배정
    f = sigmoid(A[0:H])                  # 망각 게이트: 버릴 비율
    g = np.tanh(A[H:2*H])                 # 새 기억 후보
    i = sigmoid(A[2*H:3*H])              # 입력 게이트: 저장할 비율
    o = sigmoid(A[3*H:4*H])              # 출력 게이트: 꺼낼 비율

    # 셀 상태 갱신: (옛 기억 × 버릴 비율) + (새 정보 × 저장 비율)
    c = f * c_prev + i * g                # ← 덧셈(+)이 핵심!

    # 은닉 상태: 셀 상태에서 선별 출력
    h = o * np.tanh(c)

    return h, c

# 테스트
D, H = 3, 2
np.random.seed(42)
Wx = np.random.randn(D, 4*H) * 0.01   # 4개 게이트 × H = 4H
Wh = np.random.randn(H, 4*H) * 0.01
b  = np.zeros(4*H)

x = np.array([1.0, 0.5, -1.0])
h_prev = np.zeros(H)
c_prev = np.zeros(H)

h, c = lstm_step(x, h_prev, c_prev, Wx, Wh, b)
print("은닉 상태 h:", h.round(4))
print("셀 상태 c:",  c.round(4))
