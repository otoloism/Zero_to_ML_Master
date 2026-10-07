# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #18
행렬식(Determinant) 쉽게 이해하기 — 넓이와 부피의 배율로 보는 det

원문(책): 01-06-02 「행렬식(Determinant) — 정사각행렬 속에 숨은 넓이와 부피」 https://wikidocs.net/439761
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 18_determinant.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 2단계 — 2×2 행렬식 ad − bc
#     손계산 18 과 np.linalg.det 결과를 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 2단계 — 2×2 행렬식 ad − bc")
print("=" * 60)
import numpy as np

A = np.array([[4, 2],
              [1, 5]])
print("det(A) =", np.linalg.det(A))   # ad-bc = 4*5 - 2*1 = 18


#======================================================================
# [2] 4단계 — 3×3 계산법 ① 사러스 법칙 검산
#     3×3 행렬식 −3 을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 4단계 — 3×3 계산법 ① 사러스 법칙 검산")
print("=" * 60)
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])
print("det(A) =", round(np.linalg.det(A), 4))   # -3.0


#======================================================================
# [3] 5단계 — 3×3 계산법 ② 여인수 전개 검산 (4×4 포함)
#     3×3 과 4×4 행렬식을 같은 함수로 계산합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 5단계 — 3×3 계산법 ② 여인수 전개 검산 (4×4 포함)")
print("=" * 60)
import numpy as np

B = np.array([[2, 3, 1],
              [0, 5, 0],
              [4, 1, 6]])
print("det(B) =", round(np.linalg.det(B), 4))   # 40.0

# 4x4 이상도 완전히 동일한 함수로 계산됩니다 — 여인수 전개가 만능인 이유
C = np.array([[1,0,2,-1],[3,0,0,5],[2,1,4,-3],[1,0,5,0]])
print("det(C) =", round(np.linalg.det(C), 4))


#======================================================================
# [4] 6단계 — 계산을 쉽게 만드는 행 연산 요령
#     행 교환(부호 반전)·행 k배(k배)·행 덧셈(불변)을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 6단계 — 계산을 쉽게 만드는 행 연산 요령")
print("=" * 60)
import numpy as np

A = np.array([[2., 1.],
              [3., 4.]])
print("원본       det(A)  =", round(np.linalg.det(A), 4))          # 5.0

swapped = A[[1, 0]]                       # 행 교환
print("행 교환    det     =", round(np.linalg.det(swapped), 4))    # -5.0

scaled = A.copy(); scaled[0] *= 4         # 1행에 4배
print("1행 4배    det     =", round(np.linalg.det(scaled), 4))     # 20.0

added = A.copy(); added[1] += 3 * added[0]  # 1행의 3배를 2행에 더함
print("행 덧셈    det     =", round(np.linalg.det(added), 4))      # 5.0 (그대로!)


#======================================================================
# [5] 7단계 — det(AB) = det(A)det(B)
#     두 값이 같은지 수치로 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[5] 7단계 — det(AB) = det(A)det(B)")
print("=" * 60)
import numpy as np

A = np.array([[2, 1], [3, 4]])
B = np.array([[1, 0], [5, 2]])

left  = np.linalg.det(A @ B)
right = np.linalg.det(A) * np.linalg.det(B)
print("det(AB)        =", round(left, 4))
print("det(A)*det(B)  =", round(right, 4))
