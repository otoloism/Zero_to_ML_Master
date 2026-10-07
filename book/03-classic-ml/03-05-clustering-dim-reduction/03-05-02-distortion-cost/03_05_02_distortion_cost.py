# -*- coding: utf-8 -*-
"""
03-05-02 최적화 목표 — 왜곡 비용함수 — (Optimization Objective - Distortion Cost Function) - 모두를 자기 반장 곁으로 모으는 데 드는 비용 계산서

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-02 최적화 목표 — 왜곡 비용함수.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def compute_distortion(X, centroids, labels):
    """
    왜곡(distortion) J 를 계산한다.
        J = (1/m) * Σ ‖ x⁽ⁱ⁾ − μ_c⁽ⁱ⁾ ‖²

    비유: 전 직원이 자기 지점까지 가는 거리를 제곱해서 더한 뒤 인원수로 나눈 값.
    """
    # ★ 이 한 줄이 μ_c⁽ⁱ⁾ 그 자체입니다.
    #   labels 는 (m,) 짜리 정수 배열, centroids 는 (K, n).
    #   centroids[labels] 하면 (m, n) 이 되어 "각 점이 속한 중심점"이 줄줄이 나옵니다.
    #   이것을 '팬시 인덱싱(fancy indexing)'이라고 부릅니다.
    my_centroid = centroids[labels]          # (m, n)

    diff = X - my_centroid                    # 각 점이 중심에서 얼마나 벗어났나

    # diff**2 : 원소별 제곱 (행렬 곱이 아님에 주의! @ 가 아니라 **)
    # sum(axis=1) : 특성 방향으로 더해서 "점별 제곱거리" (m,) 을 만든다
    # mean()      : m 개를 평균 → 스칼라 하나
    return np.mean(np.sum(diff ** 2, axis=1))

def fit_kmeans_with_trace(X, n_clusters, max_iter=100, seed=42):
    """할당 직후 / 이동 직후에 각각 J 를 기록하며 학습한다."""
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), n_clusters, replace=False)].copy()
    trace = []                                # (단계이름, J) 를 쌓아둘 리스트

    for step in range(max_iter):
        # ── ① 할당 단계 : μ 고정, c 최소화 ───────────────────
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        trace.append(("assign", compute_distortion(X, centroids, labels)))

        # ── ② 이동 단계 : c 고정, μ 최소화 ───────────────────
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(n_clusters)
        ])
        # ↑ 리스트 컴프리헨션 : for 문을 한 줄로 접은 것.
        #   'A if 조건 else B' 는 삼항 연산자로, 빈 클러스터면 옛 중심을 유지합니다.
        trace.append(("move", compute_distortion(X, new_centroids, labels)))

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, labels, trace

# ── 실행 ────────────────────────────────────────────────
X, _ = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

centroids, labels, trace = fit_kmeans_with_trace(X, 3)

for i, (kind, J) in enumerate(trace):
    print(f"step{i}  {kind:6s}  J = {J:.6f}")

# sklearn 의 inertia_ 와 비교
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("내 J        :", round(compute_distortion(X, centroids, labels), 6))
print("내 J × m    :", round(compute_distortion(X, centroids, labels) * len(X), 4))
print("km.inertia_ :", round(km.inertia_, 4))
