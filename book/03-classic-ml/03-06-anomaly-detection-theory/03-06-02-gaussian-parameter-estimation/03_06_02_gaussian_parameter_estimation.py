# -*- coding: utf-8 -*-
"""
03-06-02 가우시안 분포와 파라미터 추정 — (Gaussian Distribution & Parameter Estimation) - 데이터에 가장 잘 맞는 종 모양 모자를 씌우는 법

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-06장 - 이상탐지 이론/03-06-02 가우시안 분포와 파라미터 추정.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
import matplotlib.pyplot as plt

def estimate_gaussian(X):
    """
    데이터로부터 μ와 σ²을 추정한다 (최대우도추정, MLE).

    비유:
    다트 자국 100개를 보고
    "이 사람은 어디를 겨눴고(μ), 손이 얼마나 떨렸나(σ)"를
    거꾸로 알아맞히는 작업입니다.

    X : (m, n) 배열 — 데이터 m개, 특성 n개
    반환 : mu (n,), sigma2 (n,)  ← 특성마다 하나씩
    """
    # axis=0 : '행 방향으로 내려가며' 계산 → 열(특성)별 값이 나옴
    #   axis=1 로 잘못 쓰면 '데이터 하나 안에서의 평균'이 되어 완전히 틀립니다!
    mu = X.mean(axis=0)

    # np.var 의 기본값은 ddof=0, 즉 1/m 로 나눕니다 → 강의 공식과 동일
    #   (ddof=1 로 주면 1/(m-1). 표본이 크면 차이가 거의 없습니다)
    sigma2 = X.var(axis=0)

    return mu, sigma2

def gaussian_pdf(x, mu, sigma2):
    """
    1차원 가우시안 확률밀도 p(x; μ, σ²) 을 계산한다.
        p(x) = 1/(√(2πσ²)) · exp( −(x−μ)² / (2σ²) )
    """
    # np.pi 는 원주율 π. np.sqrt 는 제곱근.
    # 앞부분(정규화 상수)과 뒷부분(지수)을 나눠 쓰면 읽기 쉽습니다.
    coef = 1.0 / np.sqrt(2 * np.pi * sigma2)
    exponent = -((x - mu) ** 2) / (2 * sigma2)
    return coef * np.exp(exponent)

# ── 앞 절에서 만든 정상 엔진 데이터 사용 ────────────────────
rng = np.random.default_rng(42)
good = rng.normal(loc=[14.0, 12.0], scale=[1.2, 1.5], size=(10000, 2))
X_train = good[:6000]        # 훈련에는 6,000개만 사용 (나머지는 CV/Test용)

mu, sigma2 = estimate_gaussian(X_train)
print("mu     =", mu.round(4))
print("sigma2 =", sigma2.round(4))
print("sigma  =", np.sqrt(sigma2).round(4))

# ── 히스토그램 위에 추정한 곡선을 겹쳐 그리기 ────────────────
# 이것이 "가정이 맞는지" 확인하는 가장 확실한 방법입니다.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
names = ['$x_1$ : heat', '$x_2$ : vibration']

for j in range(2):
    # density=True : 막대 높이를 '개수'가 아니라 '밀도'로 → 곡선과 축이 맞음
    axes[j].hist(X_train[:, j], bins=50, density=True,
                 alpha=0.6, color='tab:blue', label='실제 데이터')

    # linspace(시작, 끝, 개수) : 구간을 균등하게 쪼갠 배열 → 곡선을 부드럽게
    grid = np.linspace(X_train[:, j].min(), X_train[:, j].max(), 300)
    axes[j].plot(grid, gaussian_pdf(grid, mu[j], sigma2[j]),
                 'r-', linewidth=2.5, label='추정한 가우시안')

    axes[j].set_title(names[j])
    axes[j].legend()

plt.tight_layout()   # 그림들이 겹치지 않게 여백 자동 조정
plt.show()
