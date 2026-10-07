# -*- coding: utf-8 -*-
"""
04-06-02 최대 가능도 추정 (MLE) — 생성 모델 학습의 원리

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-02 최대 가능도 추정 (MLE) — 생성 모델 학습의 원리.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 직관
import numpy as np  # NumPy: 배열/행렬 연산 핵심 라이브러리

def normal_pdf(x, mu, sigma):
    # 정규분포 PDF 공식을 그대로 코드로 옮긴 것
    # 1) 정규화 상수: 전체 넓이를 1로 맞춤
    norm_const = 1 / np.sqrt(2 * np.pi * sigma**2)
    # 2) 지수 부분: 평균에서 멀어질수록 급격히 작아짐
    exponent = -(x - mu)**2 / (2 * sigma**2)
    return norm_const * np.exp(exponent)

x = np.array([-2,-1,0,1,2])
print(normal_pdf(x, mu=0, sigma=1).round(4))


# %% [Block 2] 🖼️ 시각화 — 정규분포 PDF 곡선
import numpy as np  # NumPy: 배열/행렬 연산 핵심 라이브러리

trials = 10000          # 실험을 1만 번 반복
sums = []
for _ in range(trials):
    dice = np.random.randint(1, 7, size=100)  # 주사위 100개
    sums.append(dice.sum())               # 100개의 합 1개 기록

sums = np.array(sums)
print("평균:", sums.mean().round(2), "(이론값: 100×3.5=350)")
print("표준편차:", sums.std().round(2))

# 68-95-99.7 규칙(00-05) 확인
mu, sigma = sums.mean(), sums.std()
within_1sigma = np.mean((sums > mu-sigma) & (sums < mu+sigma))
print("μ±σ 안의 비율:", within_1sigma.round(3), "(이론값 0.68)")


# %% [Block 3] 📐 공식
np.random.seed(0)
X = np.random.normal(0, 1, 100000)    # N(0,1)
Y = np.random.normal(0, 1, 100000)    # N(0,1)
Z = X + Y

print("Z의 평균:", Z.mean().round(3), "(이론값 0)")
print("Z의 분산:", Z.var().round(3), "(이론값 1+1=2)")


# %% [Block 4] 구현 관점 — 경사 상승(Gradient Ascent)
import numpy as np

def neg_log_likelihood(params, data):
    """
    음의 로그 우도(NLL)를 계산합니다.
    이 값을 최소화하는 것이 MLE입니다.
    """
    mu, log_sigma = params           # log_sigma: sigma>0 제약을 자동으로 만족시키는 트릭
    sigma = np.exp(log_sigma)
    n = len(data)
    # 로그 우도 공식을 그대로 구현
    ll = -0.5 * n * np.log(2*np.pi*sigma**2) \
         - np.sum((data-mu)**2) / (2*sigma**2)
    return -ll   # 최대화 -> 최소화로 부호 반전

# 실제 100개 샘플 생성 (참값: mu=5, sigma=2)
np.random.seed(42)
data = np.random.normal(5, 2, 100)

# 해석적 해 (공식으로 바로 계산 - 5단계에서 유도한 식)
mu_hat = data.mean()
sigma_hat = data.std()
print("MLE 추정 μ̂:", mu_hat.round(3), "(참값 5)")
print("MLE 추정 σ̂:", sigma_hat.round(3), "(참값 2)")
