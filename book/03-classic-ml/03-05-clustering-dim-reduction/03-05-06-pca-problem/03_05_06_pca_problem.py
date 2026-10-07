# -*- coding: utf-8 -*-
"""
03-05-06 차원 축소와 PCA 문제 정의 — (Dimensionality Reduction - Problem Formulation) - 그림자를 가장 잘 드리우는 각도 찾기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-06 차원 축소와 PCA 문제 정의.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np

# 두 특성이 서로 강하게 비례하는 6개 점 (cm-inch 상황을 흉내낸 데이터)
X = np.array([[2., 1.],
              [3., 3.],
              [4., 3.],
              [5., 5.],
              [6., 5.],
              [4., 4.]])

# ── ① 중심화 : 평균을 빼서 원점 기준으로 옮긴다 ────────────
#    axis=0 은 "행 방향으로 내려가며" 평균 → 열(특성)별 평균이 나옴
mu = X.mean(axis=0)
Xc = X - mu                      # 브로드캐스팅: (6,2) - (2,) → 각 행에서 빼줌
print("평균:", mu)

# ── ② 공분산 행렬 : 특성들이 함께 움직이는 정도 ────────────
#    len(X)-1 로 나누는 것은 '표본 공분산'(자유도 보정, 01-12 참조)
Sigma = Xc.T @ Xc / (len(X) - 1)   # @ 는 행렬 곱셈 기호
print("공분산 행렬:\n", Sigma)

# ── ③ 고유값 분해 : 가장 넓게 퍼지는 방향 찾기 ─────────────
#    eigh 는 '대칭 행렬 전용' 고유값 함수 (공분산은 항상 대칭이라 안전+빠름)
#    반환값은 오름차순이므로 [::-1] 로 뒤집어 큰 것부터 놓습니다.
eigvals, eigvecs = np.linalg.eigh(Sigma)
order = np.argsort(eigvals)[::-1]      # 내림차순 정렬 인덱스
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
print("고유값:", eigvals)

u1 = eigvecs[:, 0]                     # ★ 제1 주성분 방향 u⁽¹⁾
print("u1:", u1, "  길이:", np.linalg.norm(u1))

# ── ④ 투영과 오차 계산 ────────────────────────────────────
z = Xc @ u1                      # 내적 → 1차원 새 좌표 (PC score)
approx = np.outer(z, u1)         # outer: z를 다시 2차원 위치로 되돌림 (z·u)
err = ((Xc - approx) ** 2).sum(axis=1)   # 점별 투영 오차 제곱

total = (Xc ** 2).sum(axis=1).mean()          # 총 분산(빗변 제곱의 평균)
print("평균 투영오차 : %.4f" % err.mean())
print("살린 정보 비율: %.4f" % (1 - err.mean() / total))

# ── ⑤ 비교 : 그냥 x₂를 버렸다면? ───────────────────────────
err_drop = (Xc[:, 1] ** 2)             # x₂를 0으로 만든 것과 같음
print("[특성 버리기] 평균 오차 : %.4f" % err_drop.mean())
print("[특성 버리기] 살린 비율 : %.4f" % (1 - err_drop.mean() / total))
