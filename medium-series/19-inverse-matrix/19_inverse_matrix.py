# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #19
역행렬(Inverse Matrix) 구하는 법 — 가우스-조던 소거법부터 (AB)⁻¹ = B⁻¹A⁻¹까지

원문(책): 01-06-03 「역행렬(Inverse Matrix) — 행렬의 나눗셈을 완성하다」 https://wikidocs.net/439760
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 19_inverse_matrix.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 3단계 — 예제① 2×2 역행렬과 검산
#     np.linalg.inv 결과에 A 를 곱해 단위행렬이 나오는지 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 3단계 — 예제① 2×2 역행렬과 검산")
print("=" * 60)
import numpy as np

A = np.array([[1., 3.],
              [2., 7.]])

A_inv = np.linalg.inv(A)
print("A의 역행렬:")
print(A_inv)

# 검산: A @ A_inv 는 반드시 단위행렬이 나와야 한다
print("\nA @ A_inv =")
print(np.round(A @ A_inv, 5))


#======================================================================
# [2] 4단계 — 예제② 3×3 역행렬: 가우스-조던 (SymPy rref)
#     첨가행렬 [B|I] 를 rref 로 바꿔 역행렬을 얻고 .inv() 와 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 4단계 — 예제② 3×3 역행렬: 가우스-조던 (SymPy rref)")
print("=" * 60)
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


#======================================================================
# [3] 5단계 — 행렬식·수반행렬 공식 (2×2 전용)
#     adj(A)/det(A) 를 직접 구현해 np.linalg.inv 와 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 5단계 — 행렬식·수반행렬 공식 (2×2 전용)")
print("=" * 60)
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


#======================================================================
# [4] 6단계 — 역행렬이 존재하지 않는 경우
#     det ≈ 0 인 행렬의 역행렬 계산을 시도합니다.
#     ⚠️ det(C) 는 실행 환경(NumPy/LAPACK)에 따라 0.0 또는 6.66e-16 같은 0에 아주 가까운 값으로 출력될 수 있습니다(부동소수점 오차).
#======================================================================
print("\n" + "=" * 60)
print("[4] 6단계 — 역행렬이 존재하지 않는 경우")
print("=" * 60)
import numpy as np

C = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])

print("det(C) =", np.linalg.det(C))   # 0에 아주 가까운 값(부동소수점 오차 포함)

try:
    C_inv = np.linalg.inv(C)
except np.linalg.LinAlgError as e:
    print("에러 발생:", e)


#======================================================================
# [5] 7단계 — 역행렬의 성질 (AB)⁻¹ = B⁻¹A⁻¹
#     두 결과가 같은지 np.allclose 로 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[5] 7단계 — 역행렬의 성질 (AB)⁻¹ = B⁻¹A⁻¹")
print("=" * 60)
import numpy as np

A = np.array([[1., 3.], [2., 7.]])
B = np.array([[2., 0.], [1., 1.]])

left  = np.linalg.inv(A @ B)             # (AB)^-1
right = np.linalg.inv(B) @ np.linalg.inv(A)   # B^-1 A^-1

print("(AB)^-1 =\n", np.round(left, 5))
print("B^-1 A^-1 =\n", np.round(right, 5))
print("두 결과가 같은가?", np.allclose(left, right))
