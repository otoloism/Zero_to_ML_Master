# -*- coding: utf-8 -*-
"""
02-07 🔢 7부 — NumPy 딥러닝의 계산 엔진

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-07 🔢 7부 — NumPy  딥러닝의 계산 엔진.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-07-Z NumPy 배열 — "숫자들의 격자판"
import numpy as np      # 관례: 반드시 np 라는 별명으로

# ── 1차원 (벡터) ──
a = np.array([1, 2, 3])
print(a, a.shape, a.ndim, a.dtype, a.size)

# ── 2차원 (행렬) ──
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(B.shape, B.ndim, B.dtype)   # (3,2) = 3행 2열

# ── 자주 쓰는 생성 함수 ──
print(np.zeros((2, 3)))        # 0으로 채운 2×3 (편향 초기화!)
print(np.ones(3))              # 1로 채운 벡터
print(np.arange(0, 10, 2))     # range의 배열 버전
print(np.linspace(0, 1, 5))    # 0~1을 5등분 (그래프 x축용!)

# ── 랜덤 (가중치 초기화) ──
rng = np.random.default_rng(0)  # 시드 고정 → 재현 가능
print(np.round(rng.normal(0, 1, (2, 3)), 2))


# %% [Block 2] 02-07-AA NumPy 심화 — 인덱싱 · reshape · 축(axis)
import numpy as np

X = np.arange(1, 13).reshape(3, 4)
print(X)

print("0번 행    :", X[0])          # 첫 줄 전체
print("1번 열    :", X[:, 1])       # : = "전부"
print("[1,2] 값  :", X[1, 2])       # 1행 2열
print("블록      :", X[0:2, 1:3].tolist())

# ── 불리언 인덱싱: 조건에 맞는 것만 (파이썬 if의 배열 버전!) ──
print("6보다 큰 값:", X[X > 6])

# ── np.where: 조건에 따라 다른 값 (ReLU의 원리!) ──
print(np.where(X > 6, 1, 0))


# %% [Block 3] 02-07-AA NumPy 심화 — 인덱싱 · reshape · 축(axis)
import numpy as np

c = np.arange(12)
print(c.reshape(3, 4))       # 3행 4열로
print(c.reshape(-1, 6))      # -1 = "알아서 계산해" → 2행 6열
print(c.reshape(3, 4).T)     # .T = 전치 (행↔열 뒤집기)

M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("전체합  :", M.sum())
print("axis=0  :", M.sum(axis=0))   # 각 열의 합
print("axis=1  :", M.sum(axis=1))   # 각 행의 합
print("평균    :", M.mean(), "최대:", M.max())
print("행별 최대 위치:", M.argmax(axis=1))  # 예측 클래스 뽑기!


# %% [Block 4] 02-07-AB 원소별 연산 vs 행렬 곱 — 신경망 계산의 심장
import numpy as np

# ── ① 원소별 연산: 같은 위치끼리 ──
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print("a + b =", a + b)      # 짝끼리 더하기
print("a * b =", a * b)      # 짝끼리 곱하기 (행렬곱 아님!)
print("a * 2 =", a * 2)      # 스칼라 곱 (전부에 2배)
print("a @ b =", a @ b)      # 벡터 내적 = 곱해서 전부 더함

# ── ② 행렬 곱: np.dot 또는 @ (둘은 같습니다) ──
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print("\nA @ B (행렬 곱):")
print(np.dot(A, B))
# [0,0] = 1×5 + 2×7 = 19    [0,1] = 1×6 + 2×8 = 22
# [1,0] = 3×5 + 4×7 = 43    [1,1] = 3×6 + 4×8 = 50

print("\nA * B (원소별 곱):")
print(A * B)              # 완전히 다른 결과!


# %% [Block 5] 02-07-AB 원소별 연산 vs 행렬 곱 — 신경망 계산의 심장
import numpy as np

x = np.array([1.0, 0.5])              # 입력 뉴런 2개
W = np.array([[0.1, 0.3, 0.5],        # 가중치 (2×3)
              [0.2, 0.4, 0.6]])
b = np.array([0.1, 0.2, 0.3])          # 편향 (출력 뉴런 3개)

y = np.dot(x, W) + b                   # 🔗 01-13의 z = xW + b !

print(f"입력 x  : {x}  shape {x.shape}")
print(f"가중치 W: shape {W.shape}")
print(f"출력 y  : {y}  shape {y.shape}")

# 손으로 검산해보기
print("\n[검산] y[0] = 1.0×0.1 + 0.5×0.2 + 0.1 =",
      round(1.0*0.1 + 0.5*0.2 + 0.1, 4))


# %% [Block 6] 02-07-AC 브로드캐스트 — 크기가 달라도 자동 확장
import numpy as np

# ── ① 스칼라 × 배열: 숫자 하나가 전체로 퍼짐 ──
A = np.array([[1, 2], [3, 4]])
print("A * 10 =")
print(A * 10)                    # 10이 (2,2) 전체로 확장

# ── ② 행렬 + 행벡터: 신경망의 편향 더하기! ──
Y = np.array([[0.2, 0.5, 0.8],    # 배치 2개 × 뉴런 3개
              [0.1, 0.3, 0.7]])
b = np.array([0.1, 0.2, 0.3])      # 편향 (뉴런 3개)

print(f"\nY shape {Y.shape} + b shape {b.shape}")
print(Y + b)                    # b가 두 행 모두에 적용!

# ── ③ 열벡터 + 행렬: 세로 방향 확장 ──
col = np.array([[1], [2]])       # shape (2, 1) — 열벡터
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"\n(2,1) + (2,3):")
print(M + col)                  # 열벡터가 가로로 확장


# %% [Block 7] 02-07-AD NumPy는 왜 빠른가 — 직접 측정해보기
import numpy as np
import time

N = 1_000_000                     # 100만 개 (숫자에 _ 를 넣어 읽기 쉽게)

py_a = list(range(N))              # 파이썬 리스트
py_b = list(range(N))
np_a = np.arange(N, dtype=np.float64)  # NumPy 배열
np_b = np.arange(N, dtype=np.float64)

# ── 파이썬 for문 방식 ──
t0 = time.perf_counter()
result_py = sum(x * y for x, y in zip(py_a, py_b))
t1 = time.perf_counter()

# ── NumPy 벡터화 방식 ──
t2 = time.perf_counter()
result_np = (np_a * np_b).sum()
t3 = time.perf_counter()

py_time, np_time = t1 - t0, t3 - t2
print(f"파이썬 for  : {py_time*1000:.1f} ms")
print(f"NumPy 벡터화: {np_time*1000:.1f} ms")
print(f"→ 약 {py_time/np_time:.0f}배 빠름")
print(f"결과 동일   : {abs(result_py - result_np) < 1}")
