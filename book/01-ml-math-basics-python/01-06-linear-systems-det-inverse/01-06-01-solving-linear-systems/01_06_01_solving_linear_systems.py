# -*- coding: utf-8 -*-
"""
1차식: x - 2y = 0 / 2차식: x^2 + y^2 = 5

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-06📐연립방정식·행렬식·역행렬/01-06-01📐 연립방정식의 의미와 풀이법 — 대수와 기하를 잇는 다리.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 🔵 2단계 — 1차-2차 연립방정식: 무조건 대입법
from sympy import symbols, Eq, solve

x, y = symbols('x y')

# 1차식: x - 2y = 0  /  2차식: x^2 + y^2 = 5
eq1 = Eq(x - 2*y, 0)
eq2 = Eq(x**2 + y**2, 5)

solutions = solve([eq1, eq2], [x, y])
print("해:", solutions)


# %% [Block 2] 🔵 3단계 — 해의 개수 ↔ 그래프의 교점: 대수를 기하로 해석하기
from sympy import symbols, Eq, solve

x, y = symbols('x y')
circle = Eq(x**2 + y**2, 1)   # 반지름 1인 원

# 경우 1: 교점 0개 — 원에서 먼 직선
line0 = Eq(y, x + 3)
print("0개 예상:", solve([line0, circle], [x, y]))

# 경우 2: 교점 1개 — 원에 접하는 직선
line1 = Eq(y, 1)
print("1개 예상:", solve([line1, circle], [x, y]))

# 경우 3: 교점 2개 — 원을 가로지르는 직선
line2 = Eq(y, x)
print("2개 예상:", solve([line2, circle], [x, y]))


# %% [Block 3] 🔴 4단계 — 2차-2차 연립방정식: 인수분해가 핵심
from sympy import symbols, Eq, solve, factor

x, y = symbols('x y')

# 2차-2차 연립방정식
eq1 = Eq(x**2 - y**2, 0)
eq2 = Eq(x**2 + y**2, 4)

# 1) eq1이 인수분해되는지 확인
print("인수분해:", factor(x**2 - y**2))   # (x-y)(x+y)

# 2) SymPy에게 통째로 풀어달라고 해도 같은 결과가 나오는지 검증
solutions = solve([eq1, eq2], [x, y])
print("해:", solutions)
