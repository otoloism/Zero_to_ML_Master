# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #15
텐서(Tensor)란 무엇인가 — 스칼라·벡터·행렬에서 다차원 배열까지, 숫자 상자의 차원 여행

원문(책): 01-05-01 「텐서(Tensor)란 뭘까 — 숫자 상자의 차원 여행」 https://wikidocs.net/439758
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 15_tensors.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 01-05-01-C — 코드로 확인하기: 0~3차원 텐서
#     스칼라·벡터·행렬·3차원 텐서의 shape 과 ndim 을 출력합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-05-01-C — 코드로 확인하기: 0~3차원 텐서")
print("=" * 60)
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


#======================================================================
# [2] 01-05-01-C — reshape: 모양만 바꾸기
#     원소 12개를 (12,) → (3, 4) 로 바꿉니다 (위 셀의 np 를 사용).
#======================================================================
print("\n" + "=" * 60)
print("[2] 01-05-01-C — reshape: 모양만 바꾸기")
print("=" * 60)
arr = np.arange(12)          # 0~11, shape (12,) 1차원 텐서
print(arr.shape)             # (12,)

reshaped = arr.reshape(3, 4) # 원소 12개를 유지한 채 모양만 (3,4)로 변경
print(reshaped.shape)        # (3, 4)
print(reshaped)


#======================================================================
# [3] 01-05-01-D — 프레임워크 관점: PyTorch 텐서와 자동미분
#     requires_grad=True 텐서로 y = Σx² 의 기울기 2x 를 구합니다.
#     ⚠️ torch 가 필요합니다. Colab에는 기본 설치되어 있습니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 01-05-01-D — 프레임워크 관점: PyTorch 텐서와 자동미분")
print("=" * 60)
import torch

x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
# requires_grad=True: 이 텐서에 대한 미분값을 기억하겠다는 표시

print(x.shape)   # torch.Size([2, 2])
print(x.dim())   # 2  (차수)

y = (x ** 2).sum()
y.backward()      # 역전파 — 텐서가 계산 그래프의 한 노드가 됨
print(x.grad)     # dy/dx, x와 똑같은 (2,2) 모양의 텐서로 반환
