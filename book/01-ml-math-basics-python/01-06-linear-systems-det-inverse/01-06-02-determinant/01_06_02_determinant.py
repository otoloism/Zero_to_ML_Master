# -*- coding: utf-8 -*-
"""
4x4 이상도 완전히 동일한 함수로 계산됩니다 — 여인수 전개가 만능인 이유

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-06📐연립방정식·행렬식·역행렬/01-06-02🧮 행렬식(Determinant) — 정사각행렬 속에 숨은 넓이와 부피.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 2단계 🟢⭐ 2×2 행렬식 — 대각선을 빼면 넓이가 나온다
import numpy as np

A = np.array([[4, 2],
              [1, 5]])
print("det(A) =", np.linalg.det(A))   # ad-bc = 4*5 - 2*1 = 18


# %% [Block 2] 4단계 🟢 3×3 계산법 ① — 대각선 법칙(사러스 법칙)
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])
print("det(A) =", round(np.linalg.det(A), 4))   # -3.0


# %% [Block 3] 5단계 🔵⭐ 3×3 계산법 ② — 여인수 전개 (모든 크기에 통하는 만능열쇠 🗝️)
import numpy as np

B = np.array([[2, 3, 1],
              [0, 5, 0],
              [4, 1, 6]])
print("det(B) =", round(np.linalg.det(B), 4))   # 40.0

# 4x4 이상도 완전히 동일한 함수로 계산됩니다 — 여인수 전개가 만능인 이유
C = np.array([[1,0,2,-1],[3,0,0,5],[2,1,4,-3],[1,0,5,0]])
print("det(C) =", round(np.linalg.det(C), 4))


# %% [Block 4] 6단계 🟢 계산을 쉽게 만드는 두 가지 요령
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


# %% [Block 5] 7단계 🔴🚀 행렬식의 성질 — det(AB) = det(A)det(B)
import numpy as np

A = np.array([[2, 1], [3, 4]])
B = np.array([[1, 0], [5, 2]])

left  = np.linalg.det(A @ B)
right = np.linalg.det(A) * np.linalg.det(B)
print("det(AB)        =", round(left, 4))
print("det(A)*det(B)  =", round(right, 4))
