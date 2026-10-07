# -*- coding: utf-8 -*-
"""
키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-06장 이미지 생성 모델의 원리/04-06-05 EM 알고리즘— 잠재 변수 학습의 일반 도구.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np


# %% [Block 1] 📖 처음부터 끝까지 상세 풀이
def gmm_pdf(x, weights, means, covs):
    """
    GMM의 확률밀도함수를 계산합니다.

    비유:
    K개의 스피커(정규분포)가 동시에 소리를 낼 때,
    각 스피커의 볼륨(pi_k)을 곱해서 모두 더한 것이
    전체 소리(GMM 밀도)입니다.
    """
    p = 0
    for pi_k, mu_k, Sigma_k in zip(weights, means, covs):
        p += pi_k * mvn_pdf(x, mu_k, Sigma_k)   # 3.1절의 함수 재사용!
    return p

# 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%
weights = [0.6, 0.4]
means = [np.array([160]), np.array([175])]
covs = [np.array([[36]]), np.array([[49]])]

for h in [155, 167, 178]:
    p = gmm_pdf(np.array([h]), weights, means, covs)
    print(f"키 {h}cm 에서의 확률밀도: {p[0]:.5f}")


# %% [Block 2] 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%
def kl_divergence_discrete(q, p, eps=1e-12):
    """
    이산 분포 q, p 사이의 KL 발산을 계산합니다.
    q, p: 같은 길이의 확률 벡터 (합이 1)
    """
    q = np.asarray(q) + eps   # 로그(0) 방지를 위한 작은 값 추가
    p = np.asarray(p) + eps
    return np.sum(q * np.log(q / p))   # 정의식을 그대로 코드로

q = np.array([0.7, 0.3])
p = np.array([0.5, 0.5])
print(f"D_KL(q||p) = {kl_divergence_discrete(q, p):.4f}")
print(f"D_KL(p||q) = {kl_divergence_discrete(p, q):.4f}")  # 비대칭 확인


# %% [Block 3] 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%
import numpy as np

def mvn_pdf(x, mu, Sigma):
    """다변량 정규분포 pdf (3.1절)."""
    d = mu.shape[0]
    diff = x - mu
    inv = np.linalg.inv(Sigma)               # 01-06 역행렬
    det = np.linalg.det(Sigma)               # 01-06 행렬식
    norm_const = 1.0 / np.sqrt((2*np.pi)**d * det)
    return norm_const * np.exp(-0.5 * diff @ inv @ diff)

def e_step(X, weights, means, covs):
    """
    E-step: 책임도 gamma[n, k] 계산.
    비유: 각 데이터마다 '어느 그룹 책임이 몇 %인지' 표를 채우는 단계.
    """
    N, K = X.shape[0], len(weights)
    gamma = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            gamma[n, k] = weights[k] * mvn_pdf(X[n], means[k], covs[k])
        gamma[n] /= gamma[n].sum()            # 정규화 → 합이 1
    return gamma

def m_step(X, gamma):
    """
    M-step: 책임도로 가중된 평균/공분산/혼합비율 갱신.
    비유: 표(gamma)를 보고 각 그룹의 '가중 평균 위치'를 다시 찾는 단계.
    """
    N, K = gamma.shape
    d = X.shape[1]
    Nk = gamma.sum(axis=0)                    # N_k = sum_n gamma[n,k]

    means = np.zeros((K, d))
    covs = np.zeros((K, d, d))
    weights = Nk / N

    for k in range(K):
        means[k] = (gamma[:, k, None] * X).sum(axis=0) / Nk[k]
        diff = X - means[k]
        covs[k] = (gamma[:, k, None, None] *
                   np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / Nk[k]
    return weights, means, covs

def log_likelihood(X, weights, means, covs):
    """수렴 확인용 로그우도 (04-06-05-D의 정의식)."""
    N = X.shape[0]
    ll = 0.0
    for n in range(N):
        p = sum(w * mvn_pdf(X[n], mu, S) for w, mu, S in zip(weights, means, covs))
        ll += np.log(p + 1e-12)
    return ll

def gmm_em(X, K, n_iter=50, seed=0):
    """GMM을 EM으로 학습합니다."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    weights = np.ones(K) / K
    means = X[rng.choice(N, K, replace=False)]        # 데이터 중 K개를 초기 평균으로
    covs = np.array([np.cov(X.T) for _ in range(K)])

    history = []
    for it in range(n_iter):
        gamma = e_step(X, weights, means, covs)        # E-step
        weights, means, covs = m_step(X, gamma)        # M-step
        history.append(log_likelihood(X, weights, means, covs))
    return weights, means, covs, history


# %% [Block 4] 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)
X1 = rng.normal(loc=160, scale=6, size=(300, 1))
X2 = rng.normal(loc=175, scale=7, size=(200, 1))
X = np.vstack([X1, X2])

# 1) 직접 구현한 EM
w, mu, cov, hist = gmm_em(X, K=2, n_iter=30)
print("직접 구현 — weights:", np.round(w, 3))
print("직접 구현 — means  :", np.round(mu.ravel(), 2))

# 2) sklearn
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
print("sklearn  — weights:", np.round(gm.weights_, 3))
print("sklearn  — means  :", np.round(gm.means_.ravel(), 2))
