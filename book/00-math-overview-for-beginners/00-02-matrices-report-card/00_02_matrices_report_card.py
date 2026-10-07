# -*- coding: utf-8 -*-
"""
00-02행렬— 성적표는 곧 행렬이다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 00권  비전공자 처음부터 — 수포자 훑어보기/00-02행렬— 성적표는 곧 행렬이다.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 00-02-D 📏 행렬의 크기(shape) 표현하기 — "몇 행 몇 열인가요?"
import numpy as np

# 학생 4명 × 과목 2개 → 4행 2열(4×2) 행렬
scores = np.array([
    [85, 90],   # 철수
    [92, 88],   # 영희
    [70, 85],   # 민수
    [60, 75]    # 지수
])

print("shape(행, 열):", scores.shape)  # (4, 2) → 행 먼저, 열 나중
print("차원 수(ndim) :", scores.ndim)   # 2 (행렬은 2차원)
print("전체 원소 수  :", scores.size)   # 8 (=4×2)


# %% [Block 2] 00-02-E 🌍 우리 주변의 행렬들 — "세상은 표로 가득 차 있다"
import numpy as np

# 5×5 흑백 이미지 (255=흰색, 0=검정) — 가운데에 십자 모양이 그려진 그림
image = np.array([
    [  0,   0, 255,   0,   0],
    [  0,   0, 255,   0,   0],
    [255, 255, 255, 255, 255],
    [  0,   0, 255,   0,   0],
    [  0,   0, 255,   0,   0]
])

print("이미지 크기(shape):", image.shape)  # (5, 5)
# 밝은 칸(255)은 ■, 어두운 칸(0)은 · 로 그려보기
for row in image:
    print("".join("■" if v > 127 else "·" for v in row))


# %% [Block 3] 00-02-Q 🔁 패턴 정리 — "행과 열을 짝지어서 내적한다"
import numpy as np

A = np.array([[1, 2], [3, 4]])  # (m=2, n=2)
B = np.array([[5, 6], [7, 8]])  # (n=2, p=2)

m, n = A.shape          # A: m행 n열
n2, p = B.shape         # B: n행 p열
assert n == n2, "내부 차원(n)이 안 맞으면 곱할 수 없음!"

C = np.zeros((m, p))    # 결과는 m×p 크기
for i in range(m):           # 결과의 행
    for j in range(p):       # 결과의 열
        for k in range(n):   # 짝지어 곱할 대상(합의 인덱스)
            C[i, j] += A[i, k] * B[k, j]   # 곱해서 누적 = 내적

print("직접 구현한 결과:\n", C)
print("NumPy @ 결과:\n", A @ B)


# %% [Block 4] 00-02-Y 🤔 역행렬은 왜 필요할까? — "행렬에는 나눗셈이 없다"
import numpy as np

# 풀고 싶은 방정식:  A x = b
A = np.array([[2., 1.],
               [1., 3.]])
b = np.array([5., 10.])

# 방법 1) 역행렬을 직접 구해서 곱하기: x = A⁻¹ b
x1 = np.linalg.inv(A) @ b

# 방법 2) 전용 solver 사용 (실무 권장) — 내부적으로 더 안정적/빠름
x2 = np.linalg.solve(A, b)

print("역행렬로 푼 x :", x1)
print("solve로 푼 x  :", x2)
print("검산 A@x = b ?:", np.allclose(A @ x2, b))
