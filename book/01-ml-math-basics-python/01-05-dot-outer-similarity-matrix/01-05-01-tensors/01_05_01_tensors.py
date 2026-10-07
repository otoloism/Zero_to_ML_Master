# -*- coding: utf-8 -*-
"""
0차원 텐서 (스칼라) — 축이 없음

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-05행렬내적·외적·유사도 행렬/01-05-01 텐서(Tensor)란 뭘까 — 숫자 상자의 차원 여행.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-05-01-C 6) 코드로 확인하기 — NumPy
import numpy as np

# 0차원 텐서 (스칼라) — 축이 없음
scalar = np.array(5)
print("스칼라:", scalar.shape, scalar.ndim)      # 축 개수(ndim)가 0

# 1차원 텐서 (벡터) — 축 1개
vector = np.array([1, 2, 3, 4])
print("벡터:", vector.shape, vector.ndim)         # (4,)  1

# 2차원 텐서 (행렬) — 축 2개, 풍선의 T_ij 처럼 방향 간 관계를 담을 수 있음
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("행렬:", matrix.shape, matrix.ndim)         # (2, 3)  2

# 3차원 텐서 — 축 3개 (예: 4x4 흑백 이미지 2장을 쌓음)
tensor3d = np.zeros((2, 4, 4))   # (이미지 개수, 높이, 너비)
print("3차원 텐서:", tensor3d.shape, tensor3d.ndim)  # (2, 4, 4)  3


# %% [Block 2] 3차원 텐서 — 축 3개 (예: 4x4 흑백 이미지 2장을 쌓음)
arr = np.arange(12)          # 0~11, shape (12,) 1차원 텐서
print(arr.shape)             # (12,)

reshaped = arr.reshape(3, 4) # 원소 12개를 유지한 채 모양만 (3,4)로 변경
print(reshaped.shape)        # (3, 4)
print(reshaped)


# %% [Block 3] 01-05-01-D 7) 프레임워크 관점 — PyTorch의 텐서
import torch

x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
# requires_grad=True: 이 텐서에 대한 미분값을 기억하겠다는 표시

print(x.shape)   # torch.Size([2, 2])
print(x.dim())   # 2  (차수)

y = (x ** 2).sum()
y.backward()      # 역전파 — 텐서가 계산 그래프의 한 노드가 됨
print(x.grad)     # dy/dx, x와 똑같은 (2,2) 모양의 텐서로 반환
