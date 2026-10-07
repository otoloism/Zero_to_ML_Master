# -*- coding: utf-8 -*-
"""
03-05-03 무작위 초기화와 지역 최적해 — (Random Initialization & Local Optima) - 첫 단추를 100번 다시 끼워보는 이유

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-03 무작위 초기화와 지역 최적해.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 3덩어리가 확실히 구분되는 "쉬운" 데이터를 만든다.
# 이렇게 쉬운 데이터에서도 사고가 나는지 보는 것이 이 실험의 목적!
X, _ = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

# ── 실험 A : 재시작 없이(n_init=1) 서로 다른 시드로 30번 ───────
inertias_random = []
for seed in range(30):
    km = KMeans(
        n_clusters=3,
        init='random',      # 데이터 점 중 무작위 K개 (강의의 방식)
        n_init=1,            # ★ 재시작 없음 = 한 방에 승부
        random_state=seed     # 시드를 바꿔 "다른 초기값"을 만든다
    ).fit(X)
    inertias_random.append(km.inertia_)

inertias_random = np.array(inertias_random)
print("[init='random', n_init=1] 30회")
print("  최솟값(성공) : %.4f" % inertias_random.min())
print("  최댓값(실패) : %.4f" % inertias_random.max())
print("  평균         : %.4f" % inertias_random.mean())

# 최솟값보다 (오차 이상으로) 큰 결과 = 지역 최적해에 빠진 횟수
# 1e-6 을 더하는 이유: 부동소수점 오차로 같은 값이 미세하게 다를 수 있어서
n_bad = (inertias_random > inertias_random.min() + 1e-6).sum()
print("  지역최적해 빠진 횟수 : %d / 30" % n_bad)

# ── 실험 B : 똑똑한 초기화(k-means++)로 동일 조건 ──────────────
inertias_pp = [
    KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=s).fit(X).inertia_
    for s in range(30)
]
print("\n[init='k-means++', n_init=1] 30회")
print("  최솟값 : %.4f" % min(inertias_pp))
print("  최댓값 : %.4f" % max(inertias_pp))
