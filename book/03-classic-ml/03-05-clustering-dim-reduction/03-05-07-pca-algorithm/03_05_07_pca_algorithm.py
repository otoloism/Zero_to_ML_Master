# -*- coding: utf-8 -*-
"""
03-05-07 PCA 알고리즘 — 전처리부터 복원까지 — (PCA Algorithm - Preprocessing, Covariance, SVD, Projection, Reconstruction) - 사진을 압축했다가 다시 펼쳐 보는 법

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-07 PCA 알고리즘 — 전처리부터 복원까지.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data                     # (150, 4) — 이번엔 4개 특성 전부 사용

# ═══ 1단계 : 전처리 (Mean normalization) ═══════════════════
mu = X.mean(axis=0)              # 열(특성)별 평균 → (4,)
Xc = X - mu                       # 브로드캐스팅으로 모든 행에서 빼기
print("평균 μ :", mu.round(4))
print("중심화 후 평균 :", Xc.mean(axis=0).round(10))  # 0이어야 정상!

# ═══ 2단계 : 공분산 행렬 ═══════════════════════════════════
# Xc.T @ Xc : (4,150)@(150,4) = (4,4)
# len(X)-1 로 나누는 것은 표본 공분산(자유도 보정).
# 강의처럼 m 으로 나눠도 '방향'은 완전히 동일합니다
# (모든 고유값이 같은 비율로 줄어들 뿐이라 순위와 비율이 안 바뀜).
Sigma = Xc.T @ Xc / (len(X) - 1)
print("Σ shape :", Sigma.shape)      # (4, 4) — m 과 무관!

# ═══ 3단계 : SVD ═══════════════════════════════════════════
# U : (4,4), 각 '열'이 주성분 방향
# S : (4,)  특잇값 = 각 방향의 분산량. ★ 이미 내림차순 정렬되어 있음
U, S, Vt = np.linalg.svd(Sigma)
print("S (분산량):", S.round(4))
print("설명 비율 :", (S / S.sum()).round(4))
print("누적 비율 :", np.cumsum(S / S.sum()).round(4))

# ═══ 4단계 : 투영 ══════════════════════════════════════════
k = 2
U_reduce = U[:, :k]               # ★ 콤마 앞은 행, 뒤는 열. '앞의 k개 열' = (4, 2)
Z = Xc @ U_reduce                 # (150,4)@(4,2) = (150,2)
                                  # 수식 z = U_reduceᵀ·x 를 데이터 전체에 한 번에 적용한 형태
print("Z shape :", Z.shape)
print("Z[:3] :\n", Z[:3].round(4))

# ═══ 복원 ══════════════════════════════════════════════════
# Z @ U_reduce.T : (150,2)@(2,4) = (150,4) → 다시 원래 공간으로
# 마지막에 mu 를 더해줘야 원래 위치로 돌아옵니다 (중심화를 되돌리기)
X_approx = Z @ U_reduce.T + mu
print("원본  [0] :", X[0].round(4))
print("복원  [0] :", X_approx[0].round(4))

# ═══ sklearn 과 대조 ═══════════════════════════════════════
pca = PCA(n_components=2).fit(X)
print("sklearn Z[:3] :\n", pca.transform(X)[:3].round(4))
print("sklearn 설명분산 :", pca.explained_variance_.round(4))
