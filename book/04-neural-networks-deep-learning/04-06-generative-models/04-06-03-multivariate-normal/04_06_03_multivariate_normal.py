# -*- coding: utf-8 -*-
"""
물고기 5마리 x (길이, 무게) 2개 특성 데이터

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-03 다변량 정규 분포 — 이미지를 위한 분포.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 이미지가 왜 다변량 데이터인가
import numpy as np  # NumPy: 배열/행렬 연산 핵심 라이브러리

# 물고기 5마리 x (길이, 무게) 2개 특성 데이터
data = np.array([
    [30, 500],
    [32, 520],
    [28, 470],
    [31, 505],
    [29, 480],
])

print("shape:", data.shape)   # (5, 2) → 5개의 샘플, 각 샘플은 2차원 벡터
print("axis=0 방향 평균:", data.mean(axis=0))  # 각 "열"(특성)별 평균 -> [30.  495.]
print("axis=1 방향 평균:", data.mean(axis=1))  # 각 "행"(샘플)별 평균

reshaped = data.reshape(2, 5)  # 원소 개수(10개)는 그대로, 배치만 재구성
print("reshape 결과 shape:", reshaped.shape)   # (2, 5)


# %% [Block 2] 직관적 의미
import numpy as np
import matplotlib.pyplot as plt

def multivariate_gaussian_pdf(pos, mu, sigma):
    """
    2차원 정규분포의 확률밀도를 계산합니다.

    비유:
    등산로 지도에서, 특정 좌표(pos)가 산 정상(mu)에서
    얼마나 "표준화된 거리"만큼 떨어져 있는지 계산해서
    그 지점의 높이(확률밀도)를 알려주는 함수입니다.

    인자:
        pos   : (..., 2) shape, 확률밀도를 계산할 좌표들
        mu    : (2,) shape, 평균 벡터
        sigma : (2,2) shape, 공분산 행렬
    """
    d = mu.shape[0]                       # 차원 수 (여기서는 2)
    sigma_inv = np.linalg.inv(sigma)      # 공분산 행렬의 역행렬 계산
    sigma_det = np.linalg.det(sigma)      # 공분산 행렬의 행렬식 계산
    norm_const = 1.0 / (np.sqrt((2*np.pi)**d * sigma_det))  # 정규화 상수

    diff = pos - mu                       # (x - mu), 평균으로부터의 차이 벡터
    # 마할라노비스 거리 제곱: diff^T @ Sigma^-1 @ diff 를 배치로 계산
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, sigma_inv, diff)
    return norm_const * np.exp(exponent)

# 그리드(격자) 생성: -5 ~ 5 범위를 100x100으로 촘촘히 나눔
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)               # (100,100) 격자 좌표
pos = np.dstack((X1, X2))                  # (100,100,2) 형태로 좌표쌍 결합

mu = np.array([0.0, 0.0])                  # 평균 벡터: 원점이 중심
sigma = np.array([[2.0, 1.5],
                   [1.5, 2.0]])            # 비대각 원소가 있는 공분산 행렬

Z = multivariate_gaussian_pdf(pos, mu, sigma)   # 모든 격자점의 확률밀도

plt.contour(X1, X2, Z, levels=8, cmap='viridis')  # 등고선 그리기
plt.xlabel('x1'); plt.ylabel('x2')
plt.title('2D Gaussian Contour (correlated)')
plt.axis('equal')
plt.show()


# %% [Block 3] 구현 관점
import numpy as np

np.random.seed(0)

# 진짜(true) 파라미터를 정해두고, 그로부터 데이터를 200개 생성
true_mu = np.array([2.0, 5.0])
true_sigma = np.array([[3.0, 1.2],
                        [1.2, 1.0]])

data = np.random.multivariate_normal(true_mu, true_sigma, size=200)  # (200, 2)

# ---- MLE로 mu, sigma 추정 ----
mu_hat = data.mean(axis=0)          # 표본평균 벡터: (2,)

diff = data - mu_hat                # 각 샘플의 평균으로부터의 편차: (200, 2)
# 외적의 평균 = diff^T @ diff / n  (모든 샘플에 대한 외적 합을 n으로 나눈 것)
sigma_hat = (diff.T @ diff) / data.shape[0]   # (2, 2)

print("실제 mu   :", true_mu)
print("추정 mu_hat:", mu_hat.round(2))
print()
print("실제 Sigma:\n", true_sigma)
print("추정 Sigma_hat:\n", sigma_hat.round(2))


# %% [Block 4] 코드 상세 분석
def sample_covariance(data):
    mu = data.mean(axis=0)      # (d,)
    diff = data - mu            # (n, d), 브로드캐스팅으로 각 행에서 mu를 뺌
    return (diff.T @ diff) / data.shape[0]   # (d, d)
