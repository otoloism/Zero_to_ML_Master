# -*- coding: utf-8 -*-
"""
04-04-09 시그모이드와 tanh 함수의미분유도

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-09 시그모이드와 tanh 함수의미분유도.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-09-A 시그모이드 미분 유도 (연쇄법칙 적용)
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv_formula(x):
    """해석적 미분: σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def numerical_deriv(f, x, eps=1e-5):
    """수치 미분: (f(x+ε) - f(x-ε)) / 2ε"""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# 여러 점에서 해석적 vs 수치 미분 비교
test_points = [-3, -1, 0, 1, 3]
print(f"{'x':>5}  {'해석적':>10}  {'수치적':>10}  {'일치?':>6}")
print("-" * 40)
for x in test_points:
    ana = sigmoid_deriv_formula(x)
    num = numerical_deriv(sigmoid, x)
    ok = "✅" if abs(ana - num) < 1e-8 else "❌"
    print(f"{x:>5}  {ana:10.6f}  {num:10.6f}  {ok:>6}")


# %% [Block 2] 04-04-09-B tanh 미분 유도
def tanh_deriv_formula(x):
    """해석적 미분: 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2

print(f"{'x':>5}  {'해석적':>10}  {'수치적':>10}  {'일치?':>6}")
print("-" * 40)
for x in [-3, -1, 0, 1, 3]:
    ana = tanh_deriv_formula(x)
    num = numerical_deriv(np.tanh, x)
    ok = "✅" if abs(ana - num) < 1e-8 else "❌"
    print(f"{x:>5}  {ana:10.6f}  {num:10.6f}  {ok:>6}")

print("\n⚠️ 기울기 소실 비교:")
print(f"  sigmoid 최대 미분: {sigmoid_deriv_formula(0)} (x=0)")
print(f"  tanh 최대 미분:    {tanh_deriv_formula(0)} (x=0)")
print("  → tanh이 4배 크므로 기울기 소실에 더 강함!")
