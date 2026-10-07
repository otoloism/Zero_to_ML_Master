# -*- coding: utf-8 -*-
"""
03-08-04 배치 · 확률적 · 미니배치 경사하강법 (Batch / Stochastic / Mini-batch Gradient Descent)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-08장 - 정규화 심화 및 대규모 학습/03-08-04 배치 · 확률적 · 미니배치 경사하강법.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 6단원 — 직접 구현해 비교하기
import numpy as np

# ── 실험용 데이터 만들기 ────────────────────────────────
rng = np.random.default_rng(0)
m = 10000                                    # 데이터 1만 개

# np.c_[a, b] : 두 배열을 '열 방향'으로 나란히 붙인다
#   np.ones(m) 은 전부 1인 열 → 절편(bias)을 위한 x0 = 1 트릭 (03-02-03)
X = np.c_[np.ones(m), rng.uniform(0, 10, m)]

theta_true = np.array([4.0, 3.0])            # 우리가 맞혀야 할 정답
y = X @ theta_true + rng.normal(0, 2, m)     # 정답 + 잡음

print("데이터 개수 :", m)
print("정답 theta  :", theta_true)

def compute_cost(theta):
    """전체 데이터에 대한 비용 J(theta)를 계산한다."""
    error = X @ theta - y
    return (error @ error) / (2 * m)         # 벡터의 내적 = 제곱합

print("시작점 theta=[0,0] 의 비용 J =", round(compute_cost(np.zeros(2)), 4))


# %% [Block 2] 6단원 — 직접 구현해 비교하기
def batch_gd(alpha=0.01, epochs=10):
    """배치 경사하강법: 전체를 다 보고 1번 갱신. epochs번 반복."""
    theta = np.zeros(2)
    for _ in range(epochs):
        # X.T @ (X @ theta - y) / m  :  전체 데이터의 평균 기울기
        gradient = X.T @ (X @ theta - y) / m
        theta = theta - alpha * gradient
    return theta, epochs          # 갱신 횟수 = epochs

def stochastic_gd(alpha=0.01, epochs=1, seed=1):
    """확률적 경사하강법: 샘플 1개마다 즉시 갱신."""
    r = np.random.default_rng(seed)
    theta = np.zeros(2)
    updates = 0
    for _ in range(epochs):
        # permutation : 0~m-1 을 무작위 순서로 섞은 배열
        #   이것이 강의 자료의 'Randomly shuffle dataset' 단계다
        for i in r.permutation(m):
            error = X[i] @ theta - y[i]          # 이 샘플 하나의 오차
            theta = theta - alpha * error * X[i] # 즉시 갱신 (1/m 합이 없다!)
            updates += 1
    return theta, updates

def minibatch_gd(alpha=0.01, b=10, epochs=1, seed=1):
    """미니배치 경사하강법: b개씩 묶어서 갱신."""
    r = np.random.default_rng(seed)
    theta = np.zeros(2)
    updates = 0
    for _ in range(epochs):
        idx = r.permutation(m)
        # range(0, m, b) : 0, b, 2b, ... 로 b씩 건너뛴다
        #   강의 자료의 'for i = 1, 11, 21, ..., 991' 과 같은 뜻
        for start in range(0, m, b):
            j = idx[start:start + b]                    # 이번 묶음 b개의 위치
            gradient = X[j].T @ (X[j] @ theta - y[j]) / len(j)
            theta = theta - alpha * gradient
            updates += 1
    return theta, updates

print("%-22s %10s %12s %10s" % ("방식", "갱신 횟수", "theta", "비용 J"))
print("-" * 62)
for name, (th, up) in [
    ("Batch (10 에폭)",      batch_gd(0.01, 10)),
    ("SGD (1 에폭)",         stochastic_gd(0.01, 1)),
    ("Mini-batch b=10 (1에폭)", minibatch_gd(0.01, 10, 1)),
]:
    print("%-22s %10d %12s %10.4f" % (name, up, np.round(th, 3), compute_cost(th)))

print()
print("정답(정규방정식) :", np.round(np.linalg.solve(X.T @ X, X.T @ y), 4))


# %% [Block 3] 결과 해석 — 무엇이 승부를 갈랐나 — 배치도 충분히 오래 돌리면 결국 도달한다 — 다만 몇 배가 걸리는지 보자
# 배치도 충분히 오래 돌리면 결국 도달한다 — 다만 몇 배가 걸리는지 보자
print("%-24s %14s %10s" % ("Batch 에폭 수", "theta", "비용 J"))
print("-" * 52)
for ep in [10, 100, 1000, 5000]:
    th, _ = batch_gd(0.01, ep)
    print("%-24d %14s %10.4f" % (ep, np.round(th, 3), compute_cost(th)))
print()
print("→ Batch가 SGD의 1 에폭 수준(J≈2.8)에 도달하려면 100 에폭이 필요하다.")
print("→ 미니배치 1 에폭 수준(J≈1.98)에는 1000 에폭이 필요하다.")
print("   즉 같은 결과를 얻는 데 데이터를 1000배 더 읽어야 한다는 뜻이다.")


# %% [Block 4] 실습 — 실습 1 정답 — b를 바꿔가며 1 에폭 후 성능 비교
# 실습 1 정답 — b를 바꿔가며 1 에폭 후 성능 비교
print("%10s %12s %14s %10s" % ("b", "갱신 횟수", "theta", "비용 J"))
print("-" * 50)
for b in [1, 10, 100, 1000, 10000]:
    th, up = minibatch_gd(alpha=0.01, b=b, epochs=1)
    print("%10d %12d %14s %10.4f" % (b, up, np.round(th, 3), compute_cost(th)))
print()
print("→ b가 작을수록 갱신이 많아 1 에폭 안에 더 멀리 간다.")
print("→ b=10000(전체)은 갱신이 단 1번뿐이라 Batch GD의 '1 에폭'과 정확히 같다.")
print("   J가 97.6이나 되는 것은 시작점(219.9)에서 겨우 한 걸음 뗐기 때문이다.")
print("→ 다만 b가 너무 작으면 벡터화 이득을 못 봐서 '실제 시간'은 오래 걸린다(실습 3).")
print("   갱신 횟수와 속도가 함께 좋은 b=10~100 구간이 실무의 선택지가 된다.")


# %% [Block 5] 실습 — 실습 2 정답 — 데이터가 '정렬되어' 있을 때 섞기의 효과
# 실습 2 정답 — 데이터가 '정렬되어' 있을 때 섞기의 효과
# x가 작은 것부터 큰 순서로 정렬된 데이터를 만든다 (현실에서 흔한 상황)
order = np.argsort(X[:, 1])        # argsort : 정렬했을 때의 '위치 순서'를 돌려준다
X_sorted, y_sorted = X[order], y[order]

def sgd_on_sorted(shuffle, alpha=0.01, seed=1):
    r = np.random.default_rng(seed)
    theta = np.zeros(2)
    idx = r.permutation(m) if shuffle else np.arange(m)   # 섞을지 말지
    for i in idx:
        error = X_sorted[i] @ theta - y_sorted[i]
        theta = theta - alpha * error * X_sorted[i]
    return theta

theta_answer = np.linalg.solve(X.T @ X, X.T @ y)     # 정규방정식으로 구한 정답
th_shuffled = sgd_on_sorted(shuffle=True)
th_ordered  = sgd_on_sorted(shuffle=False)

# 파라미터가 정답에서 얼마나 떨어졌는지를 L2 거리로 잰다 (03-08-01에서 배운 자!)
def distance(th):
    return np.linalg.norm(th - theta_answer)

print("정답            :", np.round(theta_answer, 3))
print("섞고 학습       :", np.round(th_shuffled, 3), " 정답과의 거리 %.3f" % distance(th_shuffled))
print("안 섞고 학습    :", np.round(th_ordered, 3), " 정답과의 거리 %.3f" % distance(th_ordered))
print()
print("→ 안 섞으면 절편이 4.0이어야 하는데 6.4로 크게 벗어났다. 거리가 10배 가까이 멀다.")
print("→ 주의: 이때 비용 J만 보면 두 경우가 비슷하게 나온다.")
print("   두 파라미터가 서로를 상쇄해 '겉보기 점수'는 비슷해지기 때문이다.")
print("   그래서 파라미터가 제대로 학습됐는지는 J가 아니라 파라미터 자체를 봐야 한다.")
print("   DataLoader(shuffle=True)가 왜 기본값인지 알 수 있다.")


# %% [Block 6] 실습 — 실습 3 정답 — 실제 실행 시간 측정
# 실습 3 정답 — 실제 실행 시간 측정
import time

print("%-26s %12s %12s" % ("방식", "실행 시간(초)", "비용 J"))
print("-" * 54)
for name, fn in [
    ("Batch (10 에폭)",        lambda: batch_gd(0.01, 10)),
    ("SGD (1 에폭)",           lambda: stochastic_gd(0.01, 1)),
    ("Mini-batch b=10",        lambda: minibatch_gd(0.01, 10, 1)),
    ("Mini-batch b=100",       lambda: minibatch_gd(0.01, 100, 1)),
]:
    t0 = time.time()               # time.time() : 현재 시각(초). 두 번 재서 빼면 걸린 시간
    th, _ = fn()
    elapsed = time.time() - t0
    print("%-26s %12.4f %12.4f" % (name, elapsed, compute_cost(th)))
print()
print("→ SGD는 파이썬 for문을 1만 번 도느라 가장 느리다(벡터화를 못 쓴다).")
print("→ 미니배치는 갱신 횟수와 속도의 균형점이다. 이것이 실무 표준인 이유다.")
