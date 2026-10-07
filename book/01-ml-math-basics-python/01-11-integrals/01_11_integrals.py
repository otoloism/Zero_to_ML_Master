# -*- coding: utf-8 -*-
"""
01-11 ∫적분— 넓이와 확률의 수학

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-11 ∫적분— 넓이와 확률의 수학.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 01-11-B 1단계 — 함수와 시그마 Σ: '더하기'를 압축하는 기호 — 수학:  Σ(i=1 to 5) i²  =  1 + 4 + 9 + 16 + 25  =  55
# 수학:  Σ(i=1 to 5) i²  =  1 + 4 + 9 + 16 + 25  =  55

total = 0                       # 결과를 담을 그릇
for i in range(1, 6):          # i = 1,2,3,4,5  ← 아래첨자~위첨자
    total = total + i**2     # 더할 대상 = i의 제곱
print(total)                    # 55


# %% [Block 2] ✅ 적분 검산 3단계 — 이것만 하면 틀릴 수 없습니다
import sympy as sp

x = sp.symbols('x')              # 기호 변수 x 선언
print(sp.integrate(3*x**2, x))      # 부정적분 → x**3
print(sp.integrate(3*x**2, (x, 1, 4)))  # 정적분 [1,4] → 63
print(sp.diff(x**3, x))            # 검산: 미분하면 3*x**2 ✅


# %% [Block 3] 01-11-J 9단계 — 코드로 적분 직접 계산해보기
import numpy as np

def f(x):
    """적분할 함수: f(x) = x^2

    비유:
    x라는 동전을 넣으면 x의 제곱이 나오는 자판기입니다.
    적분에서 이 함수는 '그 자리의 높이'를 알려주는 역할만 합니다."""
    return x**2

a, b = 0, 3                     # 적분 구간 [a, b]
exact = b**3/3 - a**3/3       # 기본정리로 구한 정확한 값 F(b)-F(a)

print("  n  |   리만 합(왼쪽)  |    오차    |    사다리꼴    |     오차")
print("-" * 68)

for n in [10, 100, 1000, 10000]:          # 조각 수를 10배씩 늘려본다
    dx = (b - a) / n                          # ① 쪼개기: 조각 하나의 폭
    x_left = np.linspace(a, b, n, endpoint=False)  # 각 조각의 왼쪽 끝점
    riemann = np.sum(f(x_left) * dx)         # ②곱하기 ③더하기 = 리만 합

    x_trap = np.linspace(a, b, n + 1)          # 사다리꼴은 양 끝점이 모두 필요
    trap = np.trapezoid(f(x_trap), x_trap)   # 양 끝 평균 → 더 빠른 수렴

    print(f"{n:>6} | {riemann:>12.6f} | {abs(riemann-exact):>10.6f} |"
          f" {trap:>12.6f} | {abs(trap-exact):>12.8f}")

print("-" * 68)
print(f"정확한 값 F(3)-F(0) = 3**3/3 - 0 = {exact:.6f}")


# %% [Block 4] 01-11-J 9단계 — 코드로 적분 직접 계산해보기
import numpy as np
from scipy import stats

# (1) 확률 = 넓이  →  CDF는 '미리 적분해 둔' 원시함수 F(x)
print("P(-1<=Z<=1) =", round(stats.norm.cdf(1) - stats.norm.cdf(-1), 6))
print("P(-2<=Z<=2) =", round(stats.norm.cdf(2) - stats.norm.cdf(-2), 6))

# (2) 기댓값 = ∫ x·p(x) dx  →  지수분포(λ=2)의 이론값은 1/λ = 0.5
x = np.linspace(0, 20, 200001)         # 0~20을 아주 잘게 쪼갬
pdf = 2 * np.exp(-2 * x)               # λe^(-λx), λ=2
print("지수분포 E[X] =", round(np.trapezoid(x * pdf, x), 6))

# (3) 몬테카를로 근사 = 딥러닝의 loss.mean()과 같은 원리
rng = np.random.default_rng(42)          # 시드 고정 → 재현 가능
samples = rng.exponential(1.0, 10000)     # λ=1에서 1만 개 표본
print("몬테카를로 E[X] =", round(samples.mean(), 6))
