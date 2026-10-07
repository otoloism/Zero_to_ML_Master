# -*- coding: utf-8 -*-
"""
x1 방향으로 2만큼 떨어진 점 vs x2 방향으로 2만큼 떨어진 점

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-04 가우스 혼합 모델(GMM) — 다봉 분포 표현.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-06-04-C 🔵 등고선으로 시각화하기 — 공분산 행렬이 만드는 타원
import numpy as np  # NumPy: 배열/행렬 연산 핵심 라이브러리

def mvn_pdf(x, mu, Sigma):
    """
    다변량 정규분포의 확률밀도함수(PDF)를 계산합니다.

    비유:
    "이 점이 이 타원 구름 안에서 얼마나 흔한 위치인가"를
    숫자 하나로 알려주는 함수입니다.
    """
    d = len(mu)
    diff = x - mu                            # 평균으로부터의 편차 벡터
    inv_Sigma = np.linalg.inv(Sigma)         # 01-06: 역행렬 -> "보정 렌즈"
    det_Sigma = np.linalg.det(Sigma)         # 01-06: 행렬식 -> 정규화 상수에 사용
    norm_const = 1 / np.sqrt((2 * np.pi) ** d * det_Sigma)
    exponent = -0.5 * diff @ inv_Sigma @ diff  # 00-01: 내적의 확장 (마할라노비스 거리)
    return norm_const * np.exp(exponent)

mu = np.array([0, 0])
Sigma_diag = np.array([[4, 0], [0, 1]])

# x1 방향으로 2만큼 떨어진 점 vs x2 방향으로 2만큼 떨어진 점
print("p(2,0) =", mvn_pdf(np.array([2, 0]), mu, Sigma_diag).round(4))
print("p(0,2) =", mvn_pdf(np.array([0, 2]), mu, Sigma_diag).round(4))


# %% [Block 2] 04-06-04-F 🔵 직접 구현하기 — GMM의 확률밀도와 책임 계산
import numpy as np

def gmm_pdf(x, pis, mus, Sigmas):
    """
    GMM의 확률밀도함수. K개 가우시안을 pi_k 가중치로 합칩니다.

    비유:
    뷔페 접시에서 각 요리(가우시안)가 차지하는 비율(pi_k)만큼
    확률을 나눠 담고 전부 합치는 것입니다.
    """
    total = 0.0
    for pi_k, mu_k, Sigma_k in zip(pis, mus, Sigmas):  # *args가 아닌 zip으로 K개를 함께 순회
        total += pi_k * mvn_pdf(x, mu_k, Sigma_k)      # 04-06-04-C의 mvn_pdf 재사용
    return total

def responsibilities(x, pis, mus, Sigmas):
    """
    데이터 x 하나에 대해, 각 성분 k가 이 데이터를 얼마나
    "책임"지는지(gamma_k)를 계산합니다. 04-06-04-E의 수식 그대로입니다.
    """
    K = len(pis)
    weighted = np.array([
        pis[k] * mvn_pdf(x, mus[k], Sigmas[k]) for k in range(K)
    ])                      # 분자: pi_k * N(x; mu_k, Sigma_k), k=0..K-1
    return weighted / weighted.sum()   # 분모로 나눠 합이 1이 되도록 정규화 (사후확률)

# 3개의 성분을 가진 가상의 GMM 파라미터
pis = [0.5, 0.3, 0.2]
mus = [np.array([0, 0]), np.array([5, 5]), np.array([-4, 3])]
Sigmas = [np.eye(2), np.eye(2) * 1.5, np.array([[2, 0.5], [0.5, 1]])]

x_test = np.array([0.5, 0.2])
print("p(x) =", round(gmm_pdf(x_test, pis, mus, Sigmas), 5))
print("책임(gamma) =", responsibilities(x_test, pis, mus, Sigmas).round(3))


# %% [Block 3] 04-06-04-G 🟢 sklearn으로 실습하기 — 2D 데이터(3개 봉우리) 학습
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# 3개의 군집(봉우리)을 가진 2D 데이터 300개 생성
X, true_labels = make_blobs(
    n_samples=300, centers=3, cluster_std=1.0, random_state=42
)

# n_components=3: "산이 3개다"라고 미리 알려줌
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X)                       # 내부적으로 EM 알고리즘(04-06-05장) 수행

print("혼합 비율 (pi):", gmm.weights_.round(2))
print("평균 (mu):\n", gmm.means_.round(2))

# 새 데이터의 소속 확률(책임) 확인
sample = np.array([[0.0, 0.0]])
print("각 성분에 대한 사후확률:", gmm.predict_proba(sample).round(3))
print("가장 확률 높은 성분:", gmm.predict(sample))


# %% [Block 4] 04-06-04-K 📝 연습문제
def responsibilities_batch(X, pis, mus, Sigmas):
    N, K = X.shape[0], len(pis)
    gamma = np.zeros((N, K))
    for n in range(N):
        gamma[n] = responsibilities(X[n], pis, mus, Sigmas)  # 행 단위로 재사용
    return gamma
