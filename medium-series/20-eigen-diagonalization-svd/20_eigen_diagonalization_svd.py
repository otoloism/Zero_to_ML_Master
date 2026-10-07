# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #20
고유값·고유벡터·대각화·SVD — Av = λv 한 줄로 시작하는 행렬 분해

원문(책): 01-07 「고유값·고유벡터·대각화·SVD」 https://wikidocs.net/439755
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 20_eigen_diagonalization_svd.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 01-07-F — 코드로 확인하기: 고유값·대각화·SVD
#     고유분해, Av = λv 검증, PDP⁻¹ 재구성, 2×3 행렬의 특잇값과 U·Vᵀ shape 을 확인합니다.
#     ⚠️ NumPy 2.5 이상에서는 np.linalg.eig 가 항상 복소수 배열을 돌려주므로 값 뒤에 +0.j 가 붙어 출력될 수 있습니다(값은 같음). 또 고유값의 순서와 고유벡터의 부호(±)는 NumPy/LAPACK 환경에 따라 책과 다를 수 있습니다 — 고유벡터는 방향만 같으면 됩니다. 이 노트북은 NumPy 2.4.4로 실행했습니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-07-F — 코드로 확인하기: 고유값·대각화·SVD")
print("=" * 60)
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
