# -*- coding: utf-8 -*-
"""
04-04-05 순환 신경망 (RNN) — 시간을 기억하는 신경망

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-05 순환 신경망 (RNN) — 시간을 기억하는 신경망.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-05-A RNN 순전파 — 코드로 한 단계씩
import numpy as np

def tanh(x):
    """tanh 활성화: 출력을 -1~1 사이로 압축"""
    return np.tanh(x)

# 설정: 입력 3차원, 은닉 2차원
D, H = 3, 2
np.random.seed(42)
Wx = np.random.randn(D, H) * 0.01   # 입력용 가중치 (3×2)
Wh = np.random.randn(H, H) * 0.01   # 은닉용 가중치 (2×2) ← RNN의 핵심!
b  = np.zeros(H)                     # 편향 (2,)

# 입력 시퀀스: 2개의 단어 벡터
x1 = np.array([1.0, 0.5, -1.0])     # t=1 입력
x2 = np.array([0.0, 1.0,  0.5])     # t=2 입력

# 시작: 은닉상태를 0으로 초기화
h0 = np.zeros(H)

# t=1: 첫 번째 단어 처리
a1 = x1 @ Wx + h0 @ Wh + b            # 현재입력 + 이전상태 + 편향
h1 = tanh(a1)                        # 활성화 → 은닉상태
print("t=1 은닉상태 h1:", h1.round(4))

# t=2: 두 번째 단어 처리 — h1이 재사용됨!
a2 = x2 @ Wx + h1 @ Wh + b            # h0 대신 h1이 들어감!
h2 = tanh(a2)
print("t=2 은닉상태 h2:", h2.round(4))
print("→ h2에는 x1과 x2의 정보가 모두 누적되어 있습니다!")


# %% [Block 2] 04-04-05-B 언어 모델(RNNLM) — GPT의 원형 — 언어 모델의 학습 데이터: 입력을 한 칸 밀면 정답!
# 언어 모델의 학습 데이터: 입력을 한 칸 밀면 정답!
corpus = "you say goodbye and i say hello .".split()

inputs  = corpus[:-1]    # 입력: 마지막 빼고
targets = corpus[1:]     # 정답: 첫째 빼고 (한 칸 밀림)

print("입력:", inputs)
print("정답:", targets)
print()
for x, t in zip(inputs, targets):
    print(f"  '{x}' 다음은 → '{t}'")
