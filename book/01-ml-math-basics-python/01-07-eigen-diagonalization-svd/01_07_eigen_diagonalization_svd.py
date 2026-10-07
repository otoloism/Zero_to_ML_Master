# -*- coding: utf-8 -*-
"""
np.linalg.eig는 (고유값 배열, 고유벡터 행렬) 튜플을 반환합니다.

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-07💎고유값·고유벡터·대각화·SVD.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-07-F 코드로 확인하기
import numpy as np

A = np.array([[2., 1.],
              [1., 2.]])

# np.linalg.eig는 (고유값 배열, 고유벡터 행렬) 튜플을 반환합니다.
# eigvecs의 "각 열"이 하나의 고유벡터입니다 (행이 아님에 주의!).
eigvals, eigvecs = np.linalg.eig(A)
print("고유값:", eigvals)
print("고유벡터(열벡터):\n", eigvecs)

# 검증: A @ v == lambda * v 인지 확인
v0 = eigvecs[:, 0]                 # 첫 번째 고유벡터
print("A @ v0     :", A @ v0)
print("lambda0*v0 :", eigvals[0] * v0)

# 대각화: A = P D P^-1
P = eigvecs
D = np.diag(eigvals)
A_reconstructed = P @ D @ np.linalg.inv(P)
print("재구성된 A:\n", A_reconstructed)

# SVD (정사각형이 아닌 행렬에도 적용 가능)
B = np.array([[3., 1., 1.],
              [1., 3., 1.]])       # 2x3, 정사각형 아님
U, S, Vt = np.linalg.svd(B)
print("특이값:", S)
print("U shape:", U.shape, "Vt shape:", Vt.shape)
