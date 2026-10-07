# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #17
연립방정식 풀이법 — 대입법과 그래프 교점으로 잇는 대수와 기하

원문(책): 01-06-01 「연립방정식의 의미와 풀이법 — 대수와 기하를 잇는 다리」 https://wikidocs.net/439759
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 17_solving_systems.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 2단계 — 1차-2차 연립방정식: 대입법 검산
#     x − 2y = 0, x² + y² = 5 를 SymPy solve 로 풉니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 2단계 — 1차-2차 연립방정식: 대입법 검산")
print("=" * 60)
from sympy import symbols, Eq, solve

x, y = symbols('x y')

# 1차식: x - 2y = 0  /  2차식: x^2 + y^2 = 5
eq1 = Eq(x - 2*y, 0)
eq2 = Eq(x**2 + y**2, 5)

solutions = solve([eq1, eq2], [x, y])
print("해:", solutions)


#======================================================================
# [2] 3단계 — 해의 개수 ↔ 그래프의 교점
#     단위원과 세 직선의 교점이 0개·1개·2개인 경우를 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 3단계 — 해의 개수 ↔ 그래프의 교점")
print("=" * 60)
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


#======================================================================
# [3] 4단계 — 2차-2차 연립방정식: 인수분해
#     x² − y² 를 인수분해하고 전체 해를 solve 로 검증합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 4단계 — 2차-2차 연립방정식: 인수분해")
print("=" * 60)
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
