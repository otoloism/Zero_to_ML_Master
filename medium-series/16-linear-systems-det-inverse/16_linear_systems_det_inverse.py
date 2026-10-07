# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #16
연립방정식·행렬식·역행렬 한 번에 — det(A) = 0이 말해 주는 것

원문(책): 01-06 「연립방정식·행렬식·역행렬」 https://wikidocs.net/439752
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 16_linear_systems_det_inverse.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 01-06-E — 코드로 확인하기: det(A)=0 이면 역행렬이 없다
#     A1, A2 의 행렬식을 구하고 A2 의 역행렬 계산 오류를 try/except 로 잡습니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-06-E — 코드로 확인하기: det(A)=0 이면 역행렬이 없다")
print("=" * 60)
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
