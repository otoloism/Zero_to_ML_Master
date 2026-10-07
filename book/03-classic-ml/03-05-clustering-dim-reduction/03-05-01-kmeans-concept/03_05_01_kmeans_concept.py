# -*- coding: utf-8 -*-
"""
03-05-01 K-평균 알고리즘 개념과 절차 — (K-Means Algorithm - Concept & Procedure) - 반을 나누고 반장을 다시 뽑기를 반복하는 학급

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-01 K-평균 알고리즘 개념과 절차.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np

# ────────────────────────────────────────────────────────────
# 1) 할당 단계 : 모든 점을 "가장 가까운 중심점"에 배정한다
# ────────────────────────────────────────────────────────────
def assign_clusters(X, centroids):
    """
    X         : (m, n) 배열  — 데이터 m개, 특성 n개
    centroids : (K, n) 배열  — 중심점 K개
    반환      : labels (m,) — 각 점이 몇 번 군집인지

    비유: 학생 m명이 조장 K명 중 가장 가까운 사람에게 걸어가는 장면.
    """
    # [핵심 문법] None(= np.newaxis)은 "축을 하나 끼워넣기"입니다.
    #   X[:, None, :]         → (m, 1, n)   각 점을 세로로 세우고
    #   centroids[None, :, :] → (1, K, n)   각 중심을 가로로 눕히면
    #   두 배열의 뺄셈이 (m, K, n)으로 자동 확장(브로드캐스팅)됩니다.
    #   → "모든 점 × 모든 중심"의 차이를 for문 없이 한 번에 만든 것!
    diff = X[:, None, :] - centroids[None, :, :]

    # axis=2 는 마지막 축(특성 n)을 따라 노름(거리)을 계산하라는 뜻.
    # 결과 distances 는 (m, K) — "i번 점에서 k번 중심까지의 거리" 표.
    distances = np.linalg.norm(diff, axis=2)

    # argmin(axis=1) : 각 행(=각 점)에서 가장 작은 값의 "열 번호"를 반환.
    # min이 아니라 argmin! 우리가 원하는 건 거리 값이 아니라 군집 번호.
    return distances.argmin(axis=1), distances

# ────────────────────────────────────────────────────────────
# 2) 이동 단계 : 각 군집의 평균 위치로 중심점을 옮긴다
# ────────────────────────────────────────────────────────────
def move_centroids(X, labels, centroids):
    """
    비유: 조장이 자기 조원들의 한가운데로 걸어가는 장면.
    """
    K = centroids.shape[0]
    new_centroids = centroids.copy()   # 원본을 건드리지 않도록 복사

    for k in range(K):
        # (labels == k) 는 True/False 배열입니다. 이것을 인덱스로 넣으면
        # True인 행만 골라냅니다. → "k번 조원만 모아라"
        members = X[labels == k]

        # [빈 클러스터 방어] 조원이 0명이면 평균을 못 냅니다(0으로 나누기!).
        # 이럴 땐 그 중심점을 그대로 두거나 아무 점 위로 옮깁니다.
        if len(members) > 0:
            # axis=0 : 행 방향(= 데이터 방향)으로 평균 → 특성별 평균이 나옴
            new_centroids[k] = members.mean(axis=0)

    return new_centroids

# ────────────────────────────────────────────────────────────
# 3) 두 단계를 번갈아 반복 = K-Means 전체
# ────────────────────────────────────────────────────────────
def fit_kmeans(X, n_clusters, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)   # 재현 가능한 난수 발생기

    # [초기화] 데이터 중에서 K개를 겹치지 않게(replace=False) 골라 중심으로 삼음.
    # 허공이 아니라 "실제 점 위"에 찍어야 빈 클러스터가 덜 생깁니다.
    idx = rng.choice(len(X), n_clusters, replace=False)
    centroids = X[idx].copy()

    for step in range(max_iter):
        labels, _ = assign_clusters(X, centroids)        # ① 할당
        new_centroids = move_centroids(X, labels, centroids)  # ② 이동

        # [종료 조건] 중심점이 (거의) 안 움직이면 수렴한 것 → 멈춤.
        # allclose 는 부동소수점 오차를 감안한 "거의 같음" 비교입니다.
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids, labels, step + 1


# %% [Block 2] 실행
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 인위적으로 3덩어리 데이터를 만든다 (정답은 알지만 모델에겐 안 알려줌)
X, _ = make_blobs(n_samples=150, n_features=2,
                  centers=3, cluster_std=0.5,
                  shuffle=True, random_state=0)

centroids, labels, n_iter = fit_kmeans(X, n_clusters=3, seed=42)
print("반복 횟수:", n_iter)
print("내 구현 중심점:")
print(np.sort(centroids, axis=0))

# scikit-learn 정답지와 비교 (sanity check)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("sklearn 중심점:")
print(np.sort(km.cluster_centers_, axis=0))
