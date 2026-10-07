# -*- coding: utf-8 -*-
"""
04-06-01 정규 분포 — 모든 생성 모델의 출발점

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-01 정규 분포 — 모든 생성 모델의 출발점.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 🟢 1단계 — 확률변수와 확률분포란?
import numpy as np

# mu(평균)=0, sigma(표준편차)=1 인 정규분포에서
# 10000개의 샘플을 뽑아본다
samples = np.random.normal(loc=0, scale=1, size=10000)

print("샘플 평균:", samples.mean().round(4))    # 0에 근접
print("샘플 표준편차:", samples.std().round(4))   # 1에 근접


# %% [Block 2] X̄n = (1/n) Σ Xi →n→∞ N(μ, σ²/n)
import numpy as np

# 주사위(1~6) 100개를 던져 합을 구하는 실험을 10000번 반복
n_dice = 100
n_trials = 10000

# shape: (10000, 100) — 반복문 대신 행렬 연산으로 한 번에!
rolls = np.random.randint(1, 7, size=(n_trials, n_dice))
sums = rolls.sum(axis=1)   # 각 시행마다 100개 합산 → (10000,)

print("합의 평균:", sums.mean().round(2))     # 이론값 100×3.5=350에 근접
print("합의 표준편차:", sums.std().round(2))


# %% [Block 3] X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²) (독립) ⟹ X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)
import numpy as np

# X ~ N(2, 1), Y ~ N(3, 4) (분산 기준)
n = 100000
X = np.random.normal(loc=2, scale=1, size=n)    # σ₁ = 1 (분산 1)
Y = np.random.normal(loc=3, scale=2, size=n)    # σ₂ = 2 (분산 4)
Z = X + Y

print("Z 평균:", Z.mean().round(4))      # 이론값: 2+3 = 5
print("Z 분산:", Z.var().round(4))       # 이론값: 1+4 = 5
print("→ 평균은 더하고, 분산도 더한다!")
