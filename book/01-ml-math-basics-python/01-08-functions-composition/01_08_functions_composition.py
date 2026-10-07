# -*- coding: utf-8 -*-
"""
01-08 함수와합성함수— 딥러닝의 구조

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-08 함수와합성함수— 딥러닝의 구조.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📊 그림 1 — 합성함수 $g(f(x))$: 상자를 통과하며 값이 변한다 — 함수 = 입력을 규칙대로 바꿔 출력하는 상자
# 함수 = 입력을 규칙대로 바꿔 출력하는 상자
def f(x):
    return 2 * x + 1        # 규칙: 2배 하고 1 더하기

def g(x):
    return x ** 2          # 규칙: 제곱하기

# 합성함수 g(f(x)) — 먼저 f, 그 결과를 g에 넣기
x = 3
print("f(3)     =", f(3))              # 3 → 7
print("g(f(3))  =", g(f(3)))           # 7 → 49
print("f(g(3))  =", f(g(3)))           # 순서를 바꾸면? 9 → 19
print("순서가 중요한가?", g(f(3)) != f(g(3)))


# %% [Block 2] 1-2 🔵⭐ 신경망 = 함수를 수백 겹 쌓은 것
import numpy as np

# 신경망 한 층 = 행렬곱 + 활성화 함수 (하나의 함수)
def layer1(x):
    return np.maximum(0, 2 * x - 1)    # 선형변환 후 ReLU

def layer2(x):
    return 3 * x + 0.5

# 3층짜리 신경망 = layer2(layer1(입력))처럼 함수를 겹겹이
x = 1.0
h1 = layer1(x)            # 1층 통과
y  = layer2(h1)           # 2층 통과
print("입력 x =", x)
print("1층 출력 =", h1)
print("최종 출력 =", y)
print("\n신경망은 결국 함수를 여러 겹 합성한 것: y = layer2(layer1(x))")


# %% [Block 3] 2-1 🔵⭐ 왜 활성화 함수가 없으면 안 되는가
import numpy as np

# 활성화 함수 없이 선형변환만 쌓으면?
A = np.array([[2, 0], [0, 3]])     # 1층 (선형)
B = np.array([[1, 1], [0, 1]])     # 2층 (선형)

합쳐진_변환 = A @ B                  # 두 층을 합치면...
print("A @ B =\n", 합쳐진_변환)
print("→ 여전히 '행렬 하나' = 선형변환 하나")
print("층을 아무리 쌓아도 직선밖에 못 그린다!")


# %% [Block 4] 📊 그림 2 — 세 활성화 함수의 모양 (전부 곡선 = 비선형)
import numpy as np

z = np.array([-2, -0.5, 0, 0.5, 2])   # 입력값들

sigmoid = 1 / (1 + np.exp(-z))         # 0~1로 눌러 담기
relu    = np.maximum(0, z)             # 음수는 0, 양수는 그대로
tanh    = np.tanh(z)                   # -1~1로 눌러 담기

print("입력 z  :", z)
print("sigmoid :", sigmoid.round(3))
print("relu    :", relu)
print("tanh    :", tanh.round(3))


# %% [Block 5] 3-1 🔵⭐ 연쇄법칙 — 합성함수를 미분하는 단 하나의 규칙 — 연쇄법칙:  dy/dx = (dy/du) × (du/dx)
# 연쇄법칙:  dy/dx = (dy/du) × (du/dx)
# y = g(f(x)),  f(x)=2x+1,  g(u)=u²
x = 3
u = 2*x + 1                    # f(x)

dg_du = 2 * u                  # g'(u) = 2u
df_dx = 2                      # f'(x) = 2
dy_dx = dg_du * df_dx          # 연쇄법칙으로 곱하기

print("u = f(x) =", u)
print("dy/du =", dg_du, ", du/dx =", df_dx)
print("dy/dx = dy/du × du/dx =", dy_dx)


# %% [Block 6] 3-1 🔵⭐ 연쇄법칙 — 합성함수를 미분하는 단 하나의 규칙 — 연쇄법칙 결과가 맞는지 '수치미분'으로 검산
# 연쇄법칙 결과가 맞는지 '수치미분'으로 검산
def y(x):
    return (2*x + 1) ** 2      # g(f(x))를 통째로

x = 3
h = 1e-6
근사기울기 = (y(x + h) - y(x - h)) / (2 * h)   # 미분의 정의

print("연쇄법칙 결과 :", 28)
print("수치미분 결과 :", round(근사기울기, 4))
print("일치하는가?", abs(근사기울기 - 28) < 0.001)


# %% [Block 7] 📊 그림 4 — 층이 깊어질수록 기울기가 0으로 (0.25ⁿ)
import numpy as np

# sigmoid의 미분은 최대 0.25 — 층마다 이 값이 곱해진다
최대기울기 = 0.25

print("역전파에서 층을 거슬러 갈수록 기울기가 곱해진다:")
for 층수 in [1, 5, 10, 20]:
    남은기울기 = 최대기울기 ** 층수
    print(f"  {층수:2d}층 통과 → 0.25^{층수} = {남은기울기:.2e}")

print("\n20층이면 기울기가 사실상 0 → 앞쪽 층은 학습이 멈춘다")


# %% [Block 8] 4-2 🔵⭐ ReLU라는 해법 — 곱해도 안 줄어드는 미분
import numpy as np

# ReLU의 미분: 양수 구간에서 항상 1 → 곱해도 안 줄어든다
def relu_미분(z):
    return (z > 0).astype(float)    # z>0이면 1, 아니면 0

z = np.array([-1, 0.5, 2, 3])
print("z         :", z)
print("ReLU 미분 :", relu_미분(z))

기울기 = 1.0
for 층 in range(20):
    기울기 *= 1.0                    # 양수 구간이면 1을 곱함
print("\n20층 통과 후 기울기 =", 기울기, "→ 소실되지 않음!")
