# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #24 · 2부
확률은 넓이다 — 적분으로 읽는 확률밀도·기댓값·KL 발산·ELBO (2부)

원문(책): 01-11 「적분 — 넓이와 확률의 수학 (2부)」 https://wikidocs.net/439757
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 24_2_integral_probability_expectation.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 01-11-J 9단계 — 리만 합과 사다리꼴 공식 비교
#     n 을 10배씩 늘리면 리만 합 오차는 1/10, 사다리꼴 오차는 1/100 로 줄어드는 것을 표로 확인합니다 (정확한 값 9).
#     ⚠️ np.trapezoid 는 NumPy 2.0 이상에서 쓸 수 있습니다(1.x 에서는 np.trapz). Colab 기본 환경에서 그대로 실행됩니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-11-J 9단계 — 리만 합과 사다리꼴 공식 비교")
print("=" * 60)
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


#======================================================================
# [2] 01-11-J 9단계 — 확률과 기댓값을 적분으로 확인
#     P(-1≤Z≤1)·P(-2≤Z≤2) 를 norm.cdf 차로, 지수분포(λ=2) 기댓값을 수치적분으로, λ=1 표본 1만 개의 평균으로 몬테카를로 근사를 계산합니다 (seed 42).
#     ⚠️ 책의 실행 결과에 붙은 (이론값 1/λ = …) 설명은 출력문이 아니라 해설이라 코드 실행 시에는 숫자만 출력됩니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 01-11-J 9단계 — 확률과 기댓값을 적분으로 확인")
print("=" * 60)
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
