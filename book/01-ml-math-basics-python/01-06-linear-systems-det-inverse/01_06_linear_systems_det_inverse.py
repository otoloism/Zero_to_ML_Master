# -*- coding: utf-8 -*-
"""
A2의 역행렬을 구하려고 하면?

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-06📐연립방정식·행렬식·역행렬.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-06-E 코드로 확인하기
import numpy as np

A1 = np.array([[2., 1.], [1., 2.]])   # 해가 유일했던 행렬
A2 = np.array([[1., 1.], [2., 2.]])   # 평행한 두 직선 (해 없음)

print("A1의 행렬식:", np.linalg.det(A1))
print("A2의 행렬식:", np.linalg.det(A2))

# A2의 역행렬을 구하려고 하면?
try:
    np.linalg.inv(A2)
except np.linalg.LinAlgError as e:
    print("A2의 역행렬 계산 오류:", e)
