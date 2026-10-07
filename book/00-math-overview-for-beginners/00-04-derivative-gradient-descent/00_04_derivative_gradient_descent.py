# -*- coding: utf-8 -*-
"""
참고 — 우리가 NumPy로 손수 만든 코드 A의 결과: 2.009904

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 00권  비전공자 처음부터 — 수포자 훑어보기/00-04미분·경사하강법— 안개 낀 산에서 내려오기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np


# %% [Block 1] 📍 1단계 — 미분(Derivative)이란 무엇인가?
def forward_diff(f, x, h=1e-4):
    """
    전진차분: (f(x+h) - f(x)) / h
    비유: 앞쪽만 보고 경사를 재는 것 — 한쪽으로 치우친 측정
    """
    return (f(x + h) - f(x)) / h          # 공식 그대로!

def numerical_diff(f, x, h=1e-4):
    """
    중심차분: (f(x+h) - f(x-h)) / (2h)
    비유: 앞뒤로 똑같이 한 걸음씩 가서 재는 것 — 좌우 오차가 상쇄됨
    """
    return (f(x + h) - f(x - h)) / (2 * h)  # 공식 그대로!

def square(x):
    """f(x) = x^2  (참값 미분: f'(x) = 2x, 따라서 f'(3) = 6)"""
    return x ** 2

# h를 바꿔가며 두 방식의 오차를 비교합니다
true_value = 6.0
for h in [1e-2, 1e-4, 1e-8]:
    fwd = forward_diff(square, 3.0, h)
    cen = numerical_diff(square, 3.0, h)
    print(f"h={h:.0e} | 전진={fwd:.8f} (오차 {abs(fwd-true_value):.1e})"
          f" | 중심={cen:.8f} (오차 {abs(cen-true_value):.1e})")


# %% [Block 2] 📍 2단계 — 미분 공식과 연쇄법칙
import math

def numerical_diff(f, x, h=1e-4):
    """중심차분 수치미분 (1단계에서 만든 함수 재사용)"""
    return (f(x + h) - f(x - h)) / (2 * h)

def check_grad(name, f, df, x):
    """
    손으로 유도한 도함수 df 가 맞는지 수치미분으로 채점합니다.

    비유: 계산기로 검산하는 것과 같습니다.
    두 값이 거의 같으면 유도가 옳다는 강력한 증거입니다.
    """
    analytic = df(x)                      # 공식으로 구한 값
    numeric  = numerical_diff(f, x)       # 정의로 구한 값

    # ★ 절대오차가 아니라 '상대오차'로 비교합니다 (아래 디버깅 포인트 참고)
    #   기울기가 1536처럼 크면 절대오차도 커지는 게 정상이기 때문입니다
    rel_error = abs(analytic - numeric) / max(1.0, abs(analytic))
    ok = "✅" if rel_error < 1e-5 else "❌"
    print(f"{ok} {name:<22} 공식={analytic:12.6f}  수치={numeric:12.6f}"
          f"  상대오차={rel_error:.1e}")

# ── 예제 ①: y=(x+1)^2 → y'=2(x+1) ───────────────────────
check_grad("(x+1)^2",
           lambda x: (x + 1) ** 2,
           lambda x: 2 * (x + 1),          # 연쇄법칙 결과
           3.0)

# ── 예제 ②: y=(3x^2+1)^4 → y'=24x(3x^2+1)^3 ─────────────
check_grad("(3x^2+1)^4",
           lambda x: (3 * x**2 + 1) ** 4,
           lambda x: 24 * x * (3 * x**2 + 1) ** 3,
           1.0)

# ── 예제 ③: 3중 합성 y=((2x+1)^2+3)^2 → y'=8((2x+1)^2+3)(2x+1)
check_grad("((2x+1)^2+3)^2",
           lambda x: ((2*x + 1)**2 + 3) ** 2,
           lambda x: 8 * ((2*x + 1)**2 + 3) * (2*x + 1),
           0.0)

# ── 예제 ④: 시그모이드 → σ'(x)=σ(x)(1-σ(x)) ──────────────
def sigmoid(x):
    """σ(x) = 1 / (1 + e^-x)  — 0~1 사이로 눌러 담는 S자 함수"""
    return 1 / (1 + math.exp(-x))

def sigmoid_grad(x):
    """σ'(x) = σ(x)(1-σ(x)) — 순전파 결과만으로 계산! (유도 ⑤단계)"""
    y = sigmoid(x)
    return y * (1 - y)

check_grad("sigmoid (x=0)", sigmoid, sigmoid_grad, 0.0)
check_grad("sigmoid (x=2)", sigmoid, sigmoid_grad, 2.0)


# %% [Block 3] 📍 2단계 — 미분 공식과 연쇄법칙
class Square(Function):
    """
    입력값을 제곱합니다.  f(x) = x^2,  f'(x) = 2x

    비유: 종이를 복사해서 두 장으로 만드는 것처럼
    같은 값을 한 번 더 곱합니다.
    """

    def forward(self, x):
        self.x = x              # 역전파에서 쓸 입력을 기억해둡니다
        return x ** 2           # 순전파: 공식 f(x)=x² 그대로

    def backward(self, gy):
        x = self.x
        gx = 2 * x * gy        # f'(x)=2x 에 연쇄법칙의 gy를 곱함
        return gx

class Sigmoid(Function):
    """
    σ(x) = 1/(1+e^-x),  σ'(x) = σ(x)(1-σ(x))

    비유: 아무리 큰 값이 들어와도 0~1 사이로 눌러 담는
    '압축 스펀지'와 같습니다.
    """

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y              # ★ 입력이 아니라 '출력'을 저장! (유도 ⑤ 덕분)
        return y

    def backward(self, gy):
        gx = gy * self.y * (1 - self.y)   # y(1-y) — 지수 재계산 불필요!
        return gx


# %% [Block 4] 📍 3단계 — 편미분과 그래디언트
import numpy as np

def numerical_gradient(f, p, h=1e-4):
    """
    다변수 함수 f 의 점 p 에서의 그래디언트를 수치적으로 구합니다.

    비유: 엘리베이터 버튼을 하나씩 눌러보는 것과 같습니다.
    한 번에 하나의 성분만 살짝 움직여서 반응을 관찰합니다.
    """
    grad = np.zeros_like(p)              # 결과를 담을 빈 벡터 (p와 같은 모양)

    for i in range(p.size):        # 성분을 하나씩 순회 (= 편미분)
        tmp = p[i]                       # 원래 값을 잠시 보관

        p[i] = tmp + h                   # i번째만 +h
        f_plus  = f(p)

        p[i] = tmp - h                   # i번째만 -h
        f_minus = f(p)

        grad[i] = (f_plus - f_minus) / (2 * h)   # 중심차분
        p[i] = tmp                       # ★ 반드시 원상복구! (안 하면 다음 성분이 오염됨)

    return grad

# ── 예제 ①: f(x,y) = x² + y² ─────────────────────────────
def f_circle(p):
    """f(x,y) = x^2 + y^2  → ∇f = (2x, 2y)"""
    return p[0]**2 + p[1]**2

point = np.array([3.0, 4.0])
formula = np.array([2*point[0], 2*point[1]])   # 공식 ∇f=(2x,2y)
numeric = numerical_gradient(f_circle, point.copy())

print("[예제①] 공식 :", formula)
print("[예제①] 수치 :", np.round(numeric, 6))
print("[예제①] 크기 :", np.linalg.norm(formula), "← 00-01장의 노름!")

# ── 예제 ②: f(x,y) = 3x² + 2xy + y³ ──────────────────────
def f_cross(p):
    """∂f/∂x = 6x+2y,  ∂f/∂y = 2x+3y²"""
    x, y = p[0], p[1]
    return 3*x**2 + 2*x*y + y**3

p2 = np.array([1.0, 2.0])
formula2 = np.array([6*1 + 2*2, 2*1 + 3*2**2])   # (10, 14)
print("[예제②] 공식 :", formula2)
print("[예제②] 수치 :", np.round(numerical_gradient(f_cross, p2.copy()), 6))


# %% [Block 5] 📍 5단계 — 코드로 구현하고 실전에 적용하기
def f(x):
    """목표 함수 f(x) = x² - 4x + 5  (최솟값: x=2 에서 f(2)=1)"""
    return x**2 - 4*x + 5

def df(x):
    """도함수 f'(x) = 2x - 4  ← 2단계에서 유도한 공식 그대로"""
    return 2*x - 4

def gradient_descent(df, x_init, lr, n_steps):
    """
    경사하강법 본체.  수식:  x ← x - η·f'(x)

    비유: 안개 속 등산객이 발밑 경사(df)를 보고
    보폭(lr)만큼 n_steps 번 내려갑니다.
    """
    x = x_init
    history = [x]                       # 이동 경로를 기록 (나중에 그래프용)

    for step in range(n_steps):
        gx = df(x)                      # ① 기울기 계산 (기울기는 관례상 g 접두)
        x = x - lr * gx               # ② ★핵심★ 수식 x ← x - η·f'(x) 그 자체
        history.append(x)               # ③ 기록

        if step < 3 or (step + 1) % 10 == 0:   # 처음 3번 + 10번마다 출력
            print(f"step {step+1:2d}: grad={gx:9.4f}  x={x:9.4f}  f(x)={f(x):9.4f}")

    return x, history

x_final, hist = gradient_descent(df, x_init=10.0, lr=0.1, n_steps=30)

print(f"\n최종 x={x_final:.6f}  f(x)={f(x_final):.6f}   (정답 x=2, f=1)")

# ★ 4단계에서 유도한 공식 e_k = (1-2η)^k · e_0 으로 결과를 예측해봅니다
#   e_0 = 10 - 2 = 8,  (1-2×0.1) = 0.8,  k = 30
predicted = 8 * (0.8 ** 30)
print(f"이론 예측 x-2 = 8*(0.8)^30 = {predicted:.6f}, 실제 = {x_final-2:.6f}")


# %% [Block 6] 📍 5단계 — 코드로 구현하고 실전에 적용하기 — 4단계에서 유도한 조건:  |1 - 2η| < 1  ⟺  0 < η < 1
# 4단계에서 유도한 조건:  |1 - 2η| < 1  ⟺  0 < η < 1
# 이 예측이 정말 맞는지 여섯 가지 학습률로 확인합니다
for lr in [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]:
    x = 10.0
    diverged = False

    for step in range(30):
        x = x - lr * df(x)
        if abs(x) > 1e10:        # 발산 감지 (안 하면 inf/overflow 경고 발생)
            diverged = True
            break

    ratio = abs(1 - 2 * lr)          # 유도한 감쇠율 |1-2η|
    result = "발산 💥" if diverged else f"{x:10.4f}"
    print(f"η={lr:<5} |1-2η|={ratio:.2f}  30스텝 후 x = {result}")


# %% [Block 7] 📍 5단계 — 코드로 구현하고 실전에 적용하기
import numpy as np

def grad_circle(p):
    """f(x,y)=x²+y² 의 그래디언트 ∇f=(2x, 2y) — 3단계에서 유도"""
    return 2 * p              # 벡터 전체에 2를 곱하면 (2x, 2y) 완성!

p = np.array([3.0, 4.0])   # 시작점 (float 필수!)
lr = 0.1

for step in range(30):
    gp = grad_circle(p)              # 그래디언트 (벡터)
    p = p - lr * gp               # ★ 1변수와 완전히 같은 한 줄! (벡터 연산)
    if step < 2 or (step + 1) % 10 == 0:
        print(f"step {step+1:2d}: p={np.round(p,6)}  f={float((p**2).sum()):.8f}")


# %% [Block 8] 📍 5단계 — 코드로 구현하고 실전에 적용하기
import numpy as np

# 00-03과 동일한 데이터: (1,2), (2,3), (3,5)
X = np.array([1.0, 2.0, 3.0])
Y = np.array([2.0, 3.0, 5.0])
m = len(X)                    # 데이터 개수

w, b = 0.0, 0.0            # 파라미터 초기화 (아무 값이나 OK)
lr = 0.1

for step in range(1, 1001):
    # ── ① 순전파: 예측하고 오차 구하기 ─────────────────
    pred = w * X + b            # ŷ = wx + b  (벡터 3개 한 번에)
    e = pred - Y                 # 잔차 e = ŷ - y

    # ── ② 역전파: 공식 ⑨를 그대로 코드로 ───────────────
    gw = (2 / m) * np.sum(e * X)   # ∂L/∂w = (2/m)Σ eᵢxᵢ
    gb = (2 / m) * np.sum(e)       # ∂L/∂b = (2/m)Σ eᵢ

    # ── ③ 업데이트: 경사하강 한 걸음 ───────────────────
    w = w - lr * gw
    b = b - lr * gb

    if step in (1, 10, 100, 500, 1000):
        mse = np.mean((w * X + b - Y) ** 2)
        print(f"step {step:4d}: w={w:.6f} b={b:.6f} MSE={mse:.6f}")

# ── ④ 00-03의 정규방정식 답과 비교 검증 ────────────────
A = np.column_stack([X, np.ones(m)])
exact = np.linalg.lstsq(A, Y, rcond=None)[0]
print("정규방정식(00-03) 정답:", np.round(exact, 6))


# %% [Block 9] 📍 6단계 — 프레임워크 연결: 자동미분 — ① PyTorch
# ══════════ ① PyTorch ══════════
import torch

x = torch.tensor(10.0, requires_grad=True)   # 미분 대상으로 등록(=녹화 시작)
optimizer = torch.optim.SGD([x], lr=0.1)         # SGD = 우리가 만든 그 경사하강법

for step in range(30):
    loss = x**2 - 4*x + 5      # 순전파 (계산 과정이 녹화됨)
    optimizer.zero_grad()          # ⚠️ 이전 기울기 초기화 (안 하면 누적!)
    loss.backward()                # 역전파 → x.grad 에 2x-4 가 자동으로 채워짐
    optimizer.step()               # x = x - lr*x.grad 를 자동 수행

print("PyTorch  :", round(x.item(), 6))    # 2.009904 — 우리 코드와 동일!

# ══════════ ② TensorFlow ══════════
import tensorflow as tf

x = tf.Variable(10.0)
opt = tf.optimizers.SGD(learning_rate=0.1)

for step in range(30):
    with tf.GradientTape() as tape:      # '테이프'에 녹화 (블랙박스 비유 그대로!)
        loss = x**2 - 4*x + 5
    grads = tape.gradient(loss, [x])       # 되감아서 기울기 추출
    opt.apply_gradients(zip(grads, [x]))    # 업데이트

print("TensorFlow:", round(float(x.numpy()), 6))

# ══════════ ③ JAX ══════════
import jax, jax.numpy as jnp

def loss_fn(x):
    """손실함수를 '순수 함수'로 정의 — JAX 스타일"""
    return x**2 - 4*x + 5

grad_fn = jax.grad(loss_fn)        # ★ 도함수 '함수'를 통째로 만들어줌!

x = 10.0
for step in range(30):
    x = x - 0.1 * grad_fn(x)   # 우리가 손으로 쓴 코드와 형태가 똑같음!

print("JAX      :", round(float(x), 6))
