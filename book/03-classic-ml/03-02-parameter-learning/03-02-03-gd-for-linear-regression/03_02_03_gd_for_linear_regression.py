# -*- coding: utf-8 -*-
"""
03-02-03 선형회귀를 위한 경사하강법 — 수식 유도 — 오차라는 언덕의 기울기를 미분으로 재는 법

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-02장 - Parameter Learning/03-02-03 선형회귀를 위한 경사하강법 - 수식 유도.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] gradients.py
import numpy as np

def compute_gradients(theta0, theta1, x, y):
    """
    선형회귀의 두 기울기를 계산합니다.

    비유:
    지금 서 있는 산비탈에서 "동서 방향 경사"와 "남북 방향 경사"를
    각각 재어 알려주는 나침반과 같습니다.
    """
    m = len(x)                    # 데이터 개수. 수식의 m 입니다.

    # ① 현재 직선으로 모든 데이터를 예측합니다.
    #    x 가 배열이므로 m 개의 예측이 한 번에 나옵니다.
    y_hat = theta0 + theta1 * x

    # ② 오차를 구합니다. 부호 순서는 (예측 − 정답) 입니다.
    #    수식의 (h(x⁽ⁱ⁾) − y⁽ⁱ⁾) 와 정확히 같습니다.
    error = y_hat - y

    # ③ θ₀ 의 기울기 = 오차의 평균
    #    수식: (1/m) Σ (h(x) − y)
    #    .sum()/m 은 .mean() 과 같으므로 mean 을 써도 됩니다.
    grad0 = error.sum() / m

    # ④ θ₁ 의 기울기 = (오차 × x) 의 평균
    #    수식: (1/m) Σ (h(x) − y)·x
    #    error * x 는 원소끼리 곱해집니다 (첫 데이터끼리, 둘째끼리, ...)
    #    x 가 큰 데이터의 오차에 더 큰 비중이 실린다는 뜻입니다.
    grad1 = (error * x).sum() / m

    # 두 값을 함께 돌려줍니다. (파이썬은 여러 값을 한꺼번에 반환할 수 있습니다)
    return grad0, grad1

# ── 검산: 수치 미분과 비교하기 ────────────────────────────
# 손으로 유도한 식이 맞는지 확인하는 가장 확실한 방법입니다.
# "θ를 아주 조금 움직였을 때 J가 얼마나 변했나"를 직접 재 봅니다.
def numeric_grad0(theta0, theta1, x, y, eps=1e-5):
    # eps 는 아주 작은 값(0.00001)입니다. '입실론'이라고 읽습니다.
    # 양쪽으로 조금씩 움직여 기울기를 근사하는 '중앙차분' 방식입니다.
    j_plus  = compute_cost(theta0 + eps, theta1, x, y)
    j_minus = compute_cost(theta0 - eps, theta1, x, y)
    return (j_plus - j_minus) / (2 * eps)


# %% [Block 2] gradient_descent_linear.py
import numpy as np

# ── 데이터 준비 ─────────────────────────────────────────
# dtype=float 을 붙이는 이유: 정수 배열로 두면 나눗셈에서
# 소수점이 잘려 엉뚱한 결과가 나올 수 있기 때문입니다.
x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([4, 7, 7, 8], dtype=float)
m = len(x)

# ── 하이퍼파라미터 ──────────────────────────────────────
# '하이퍼파라미터'란 학습으로 배우는 값이 아니라
# 우리가 미리 정해 주는 설정값을 말합니다.
theta0, theta1 = 0.0, 0.0   # start with θ₀=0, θ₁=0
alpha = 0.1                    # learning rate (보폭)
n_iters = 1000                # 반복 횟수

# ── 학습 루프 : repeat until convergence ────────────────
for it in range(n_iters):

    # ① 예측 : 현재 직선으로 전체 데이터를 한 번에 예측
    y_hat = theta0 + theta1 * x

    # ② 오차 : (예측 − 정답)
    error = y_hat - y

    # ③ 기울기 두 개를 "지금의 θ" 로 각각 계산합니다.
    #    아직 θ 를 건드리지 않았다는 점이 중요합니다.
    grad0 = error.sum() / m           # ∂J/∂θ₀
    grad1 = (error * x).sum() / m     # ∂J/∂θ₁  ← x 가 곱해짐!

    # ④ 동시 갱신 (simultaneous update)
    #    오른쪽을 모두 계산한 뒤 왼쪽에 넣으므로,
    #    이 한 줄이 곧 '동시 갱신'입니다.
    theta0, theta1 = theta0 - alpha * grad0, theta1 - alpha * grad1

    # ⑤ 100번마다 상태를 출력해 J 가 줄어드는지 확인합니다.
    #    % 는 나머지 연산자입니다. it가 100의 배수일 때만 True 가 됩니다.
    if it % 100 == 0:
        J = (error ** 2).sum() / (2 * m)
        print(f"iter {it:4d}  J={J:.6f}  θ₀={theta0:.4f}  θ₁={theta1:.4f}")

print("최종:", theta0, theta1)


# %% [Block 3] multivariate_gd.py
import numpy as np

# ── 데이터 (집 4채, 특징 4개) ──────────────────────────
# 각 줄이 집 한 채입니다. "행 = 샘플, 열 = 특징"이 머신러닝의 약속입니다.
X_raw = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [1534, 3, 2, 30],
    [ 852, 2, 1, 36],
], dtype=float)

y = np.array([460, 232, 315, 178], dtype=float)

m, n = X_raw.shape   # shape 은 (행 개수, 열 개수) 를 돌려줍니다 → m=4, n=4

# ── x₀ = 1 열을 맨 앞에 붙이기 ─────────────────────────
# np.ones((m,1)) : 1로 가득 찬 m행 1열 배열을 만듭니다.
# np.hstack     : 배열을 '가로로(horizontal)' 이어 붙입니다.
# 결과: X 는 (4, 5) 모양이 되고, 첫 열이 전부 1 입니다.
X = np.hstack([np.ones((m, 1)), X_raw])

# ── 파라미터 초기화 ─────────────────────────────────────
# θ 는 특징 개수 + 1 개 필요합니다 (x₀ 몫까지).
# np.zeros 는 0으로 채운 배열을 만듭니다 → "start with θ = 0"
theta = np.zeros(n + 1)

alpha = 1e-8       # 주의: 특징 범위가 제각각이라 아주 작아야 합니다. 5단원 참고!
n_iters = 1000

for it in range(n_iters):
    # ① 예측 : h(X) = Xθ
    #    @ 는 행렬곱 기호입니다. (4,5) @ (5,) → (4,) 모양이 나옵니다.
    #    집 4채의 예측이 한 번에 계산됩니다.
    y_hat = X @ theta

    # ② 오차
    error = y_hat - y

    # ③ 모든 기울기를 한 번에 계산합니다.
    #    X.T 는 X의 전치(행과 열을 뒤집기)입니다. (5,4) 모양.
    #    X.T @ error → (5,) : j번째 원소가 Σ(h−y)·xⱼ 가 됩니다.
    #    즉 반복문 없이 n+1 개의 편미분을 동시에 얻습니다!
    grads = (X.T @ error) / m

    # ④ 동시 갱신 : 배열 연산이라 모든 θⱼ 가 한꺼번에 바뀝니다.
    theta = theta - alpha * grads

print("θ =", theta)


# %% [Block 4] feature_scaling.py
import numpy as np

def normalize(X_raw):
    """
    각 특징(열)을 평균 0, 표준편차 1 로 맞춥니다.

    비유:
    서로 다른 단위(mm, kg, 년)로 적힌 성적표를
    모두 '표준 점수'로 바꾸어 공정하게 비교하는 것과 같습니다.
    """
    # axis=0 은 "열 방향으로 계산하라"는 뜻입니다.
    # 즉 특징마다 따로 평균을 냅니다. (넓이 평균, 방 개수 평균, ...)
    # axis=1 로 하면 '집 한 채의 특징들 평균'이 되어 완전히 틀립니다!
    mu = X_raw.mean(axis=0)    # μ : 각 특징의 평균
    s  = X_raw.std(axis=0)     # s : 각 특징의 표준편차(퍼진 정도)

    # 표준편차가 0인 열은 모든 값이 같다는 뜻입니다.
    # 그대로 나누면 0으로 나누기가 되어 nan 이 발생하므로,
    # 그런 열만 1로 바꿔 나눗셈을 무력화합니다.
    s[s == 0] = 1

    # (값 − 평균) / 표준편차
    # X_raw 는 (m, n), mu 와 s 는 (n,) 모양인데도 계산이 됩니다.
    # NumPy 가 모든 행에 자동으로 맞춰 주기 때문입니다(브로드캐스팅).
    X_norm = (X_raw - mu) / s

    # mu 와 s 도 함께 돌려줍니다.
    # 나중에 새로운 집을 예측할 때 "같은 기준"으로 변환해야 하기 때문입니다.
    return X_norm, mu, s

# ── 사용 ────────────────────────────────────────────────
X_norm, mu, s = normalize(X_raw)

print("평균:", X_norm.mean(axis=0).round(6))
print("표준편차:", X_norm.std(axis=0).round(6))

# ⚠️ 새 데이터를 예측할 때는 반드시 "훈련 때의" mu, s 를 써야 합니다.
#    새 데이터로 평균을 다시 내면 기준이 달라져 예측이 망가집니다.
x_new_norm = (x_new - mu) / s


# %% [Block 5] 다. 구현 문제 — 벡터화 방식
# 벡터화 방식
grads_fast = (X.T @ error) / m

# 반복문 방식 (수식 그대로)
grads_slow = np.zeros(X.shape[1])
for j in range(X.shape[1]):    # 특징마다
    total = 0.0
    for i in range(m):            # 데이터마다
        total += error[i] * X[i, j]
    grads_slow[j] = total / m

# allclose : 부동소수점 오차를 감안해 "거의 같은지" 확인합니다.
print(np.allclose(grads_fast, grads_slow))   # True
