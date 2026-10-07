# -*- coding: utf-8 -*-
"""
04-04-11 GRU—LSTM의 경량 변형

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-11 GRU—LSTM의 경량 변형.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-11-A GRU 순전파 구현 & LSTM과 비교
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gru_step(x, h_prev, Wx, Wh, b):
    """GRU 1단계 순전파
    비유: '피처폰 한 틱' — 리셋(r)과 업데이트(z) 두 버튼만으로 동작"""
    H = h_prev.shape[0]

    # 3개 게이트의 입력을 한 번에 (LSTM은 4개, GRU는 3개!)
    A = x @ Wx + h_prev @ Wh + b         # (3H,)

    # 리셋 게이트: "과거 기억을 얼마나 무시할까?"
    r = sigmoid(A[0:H])

    # 업데이트 게이트: "과거 vs 새 정보, 비율은?"
    z = sigmoid(A[H:2*H])

    # 새 후보: 리셋된 과거 + 현재 입력으로 생성
    # ⚠️ 여기서 h_prev에 r을 곱해 "기억 리셋" 적용
    h_hat = np.tanh(x @ Wx[0:, 2*H:3*H] + (r * h_prev) @ Wh[0:, 2*H:3*H] + b[2*H:3*H])

    # 최종 은닉 상태: 시소 원리!
    # z≈1 → 과거 유지 / z≈0 → 새 정보로 교체
    h = z * h_prev + (1 - z) * h_hat

    return h

# LSTM vs GRU 파라미터 수 비교
D, H = 128, 256
lstm_params = 4 * (D + H) * H + 4 * H  # 4게이트
gru_params  = 3 * (D + H) * H + 3 * H  # 3게이트 (r, z, h̃)

print(f"입력 D={D}, 은닉 H={H}")
print(f"LSTM 파라미터: {lstm_params:>,}개")
print(f"GRU  파라미터: {gru_params:>,}개")
print(f"GRU가 {(lstm_params - gru_params) / lstm_params * 100:.1f}% 적음!")
