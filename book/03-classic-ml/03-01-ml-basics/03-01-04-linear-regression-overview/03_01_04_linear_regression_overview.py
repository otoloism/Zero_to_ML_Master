# -*- coding: utf-8 -*-
"""
03-01-04 선형회귀분석 개요 — 점들 사이로 가장 곧은 길을 찾는 자

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-01장 - 머신러닝 기초/03-01-04 선형회귀분석 개요.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] hypothesis.py
import numpy as np   # 숫자 배열 도구. 관례적으로 np 라는 별명을 붙입니다

# ── 가설 함수 ────────────────────────────────────────
# 수식 h(x) = θ₀ + θ₁·x 를 그대로 옮긴 것입니다.
# theta0 = 절편(택시 기본요금), theta1 = 기울기(km당 요금)
def hypothesis(theta0, theta1, x):
    # x 가 배열이면 모든 데이터의 예측이 "한 번에" 계산됩니다.
    # for 문이 없다는 점에 주목하세요 (벡터화, vectorization)
    return theta0 + theta1 * x

# ── 사용 예 ──────────────────────────────────────────
x = np.array([0, 1, 2, 3])   # 입력 4개

# 아직 정답을 모르니 아무 값이나 넣어 봅니다
print(hypothesis(2, 2, x))   # θ₀=2, θ₁=2 → h(x) = 2 + 2x


# %% [Block 2] 다. 구현
def predict_and_error(theta0, theta1, x, y):
    y_hat = theta0 + theta1 * x        # 예측
    error = y_hat - y                  # 오차 (예측 − 정답)
    return y_hat, (error ** 2).sum()   # 예측값과 오차제곱합
