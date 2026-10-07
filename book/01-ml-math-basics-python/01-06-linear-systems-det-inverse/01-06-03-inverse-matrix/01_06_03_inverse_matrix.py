# -*- coding: utf-8 -*-
"""
검산: A @ A_inv 는 반드시 단위행렬이 나와야 한다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-06📐연립방정식·행렬식·역행렬/01-06-03🔑 역행렬(Inverse Matrix) — 행렬의 나눗셈을 완성하다.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 🟢 3단계 — 예제① 2×2 행렬의 역행렬 구하기
import numpy as np

A = np.array([[1., 3.],
              [2., 7.]])

A_inv = np.linalg.inv(A)
print("A의 역행렬:")
print(A_inv)

# 검산: A @ A_inv 는 반드시 단위행렬이 나와야 한다
print("\nA @ A_inv =")
print(np.round(A @ A_inv, 5))


# %% [Block 2] 🔵 4단계 — 예제② 3×3 행렬의 역행렬 구하기
from sympy import Matrix

B = Matrix([[1, 2, 3],
            [0, 1, 4],
            [5, 6, 0]])

# 첨가행렬 [B|I]를 만들고 기약행사다리꼴(rref)로 바꾸면
# 왼쪽이 자동으로 단위행렬이 되고 오른쪽에 역행렬이 남는다
I = Matrix.eye(3)
augmented = B.row_join(I)
reduced, _ = augmented.rref()   # 가우스-조던의 결과 = 기약행사다리꼴

B_inv_from_rref = reduced[:, 3:]   # 오른쪽 절반만 잘라내기
print("가우스-조던으로 구한 역행렬:")
print(B_inv_from_rref)

# SymPy가 제공하는 .inv()로도 같은 결과가 나오는지 검산
print("\n.inv()로 구한 역행렬:")
print(B.inv())


# %% [Block 3] 🔵 5단계 — 역행렬 구하기② 행렬식·수반행렬 공식 (2×2 전용)
import numpy as np

def inverse_2x2(A):
    """2x2 행렬 전용 - 행렬식과 수반행렬 공식으로 역행렬 계산"""
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    det = a * d - b * c          # 행렬식
    if det == 0:
        raise ValueError("행렬식이 0이라 역행렬이 존재하지 않습니다")
    adjugate = np.array([[d, -b], [-c, a]])   # 수반행렬
    return adjugate / det

A = np.array([[1., 3.], [2., 4.]])
print("공식으로 구한 역행렬:")
print(inverse_2x2(A))
print("\nnp.linalg.inv 결과와 비교:")
print(np.linalg.inv(A))


# %% [Block 4] 🔴 6단계 — 역행렬이 존재하지 않는 경우
import numpy as np

C = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])

print("det(C) =", np.linalg.det(C))   # 0에 아주 가까운 값(부동소수점 오차 포함)

try:
    C_inv = np.linalg.inv(C)
except np.linalg.LinAlgError as e:
    print("에러 발생:", e)


# %% [Block 5] 🔵 7단계 — 역행렬의 성질
import numpy as np

A = np.array([[1., 3.], [2., 7.]])
B = np.array([[2., 0.], [1., 1.]])

left  = np.linalg.inv(A @ B)             # (AB)^-1
right = np.linalg.inv(B) @ np.linalg.inv(A)   # B^-1 A^-1

print("(AB)^-1 =\n", np.round(left, 5))
print("B^-1 A^-1 =\n", np.round(right, 5))
print("두 결과가 같은가?", np.allclose(left, right))
