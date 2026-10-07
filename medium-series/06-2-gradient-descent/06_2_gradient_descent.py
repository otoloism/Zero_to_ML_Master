# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #06 · 2부
경사하강법 완전 정복 — 안개 낀 산에서 가장 낮은 곳 찾기 (2부)

원문(책): 00-04 「미분·경사하강법 — 안개 낀 산에서 내려오기 (2부)」 https://wikidocs.net/439840
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 06_2_gradient_descent.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 5단계 — 경사하강법 구현과 이론 예측
#     f(x)=x²−4x+5 를 lr=0.1 로 30스텝 내려가고, 유도한 공식 e_k=(1−2η)^k·e_0 의 예측과 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 5단계 — 경사하강법 구현과 이론 예측")
print("=" * 60)
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


#======================================================================
# [2] 5단계 — 학습률 실험: 수렴·진동·발산
#     여섯 가지 학습률로 |1−2η| < 1 조건이 실제로 맞는지 확인합니다 (바로 위 셀의 df 를 사용).
#======================================================================
print("\n" + "=" * 60)
print("[2] 5단계 — 학습률 실험: 수렴·진동·발산")
print("=" * 60)
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


#======================================================================
# [3] 5단계 — 2변수 경사하강법
#     f(x,y)=x²+y² 에서 벡터 연산 한 줄 p = p − lr·∇f 로 원점까지 내려갑니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 5단계 — 2변수 경사하강법")
print("=" * 60)
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


#======================================================================
# [4] 5단계 — 선형회귀를 경사하강법으로 학습
#     (1,2),(2,3),(3,5) 데이터로 w, b 를 학습하고 00-03 정규방정식(lstsq) 답과 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 5단계 — 선형회귀를 경사하강법으로 학습")
print("=" * 60)
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


#======================================================================
# [5] 6단계 — 프레임워크 연결: PyTorch·TensorFlow·JAX 자동미분
#     같은 문제를 세 프레임워크의 자동미분으로 풀어 모두 2.009904 가 나오는지 확인합니다.
#     ⚠️ torch, tensorflow, jax 가 필요합니다(설치 용량이 큽니다). Colab에는 기본 설치되어 있습니다. TensorFlow/JAX 는 첫 import 시 몇 초가 걸리고, GPU가 없으면 CPU 관련 안내 로그가 출력될 수 있습니다(정상). 세 프레임워크 모두 기본이 float32 라서 결과가 책의 2.009904 대신 2.009903 처럼 마지막 자리가 다를 수 있습니다.
#======================================================================
print("\n" + "=" * 60)
print("[5] 6단계 — 프레임워크 연결: PyTorch·TensorFlow·JAX 자동미분")
print("=" * 60)
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
