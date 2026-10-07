# -*- coding: utf-8 -*-
"""
04-01-03 신경망 학습 —손실 함수와 경사 하강법

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-01장 딥러닝 이론과 구현/04-01-03 신경망 학습 —손실 함수와 경사 하강법.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-01-03-B 손실 함수 ① MSE — "(예측-정답)²의 평균"
import numpy as np

def mean_squared_error(y, t):
    """평균제곱오차: (예측-정답)²의 평균.
    비유: '과녁과 화살 사이 거리의 제곱 평균'"""
    return np.mean((y - t) ** 2)        # (차이)² → 평균

# ── 예제: 정답은 "2번 클래스" ──
t = np.array([0, 0, 1, 0, 0])             # 원-핫 정답: 2번이 1

# 좋은 예측: 2번에 높은 확률
y_good = np.array([0.1, 0.05, 0.8, 0.0, 0.05])
# 나쁜 예측: 0번에 높은 확률
y_bad = np.array([0.8, 0.05, 0.1, 0.0, 0.05])

print(f"좋은 예측 MSE: {mean_squared_error(y_good, t):.4f}")
print(f"나쁜 예측 MSE: {mean_squared_error(y_bad, t):.4f}")


# %% [Block 2] 04-01-03-C 손실 함수 ② 교차 엔트로피(CEE) — "정답의 확률에 -log"
def cross_entropy_error(y, t):
    """교차 엔트로피: -log(정답 확률).
    비유: '정답을 맞출 자신감이 낮을수록 큰 벌점'"""
    delta = 1e-7                         # log(0) 방지용 아주 작은 수
    return -np.sum(t * np.log(y + delta))

# ── 비교 ──
t = np.array([0, 0, 1, 0, 0])             # 정답: 2번

y_good = np.array([0.1, 0.05, 0.8, 0.0, 0.05])
y_bad = np.array([0.8, 0.05, 0.1, 0.0, 0.05])

print(f"좋은 예측 CEE: {cross_entropy_error(y_good, t):.4f}")
print(f"나쁜 예측 CEE: {cross_entropy_error(y_bad, t):.4f}")
print(f"\n-log(0.8) = {-np.log(0.8):.4f}")  # 높은 확률 → 작은 손실
print(f"-log(0.1) = {-np.log(0.1):.4f}")  # 낮은 확률 → 큰 손실!


# %% [Block 3] 04-01-03-D 미니배치 — "냄비 전체 대신 한 숟갈만 맛보기"
import numpy as np

# ── 가짜 데이터로 미니배치 시연 ──
total_data = 60000                       # MNIST 전체 데이터 수
batch_size = 100                         # 한 번에 뽑을 개수

# 0~59999 중 랜덤으로 100개 인덱스 뽑기
batch_idx = np.random.choice(total_data, batch_size)

print(f"전체 데이터: {total_data}장")
print(f"미니배치 크기: {batch_size}장")
print(f"선택된 인덱스 (처음 10개): {batch_idx[:10]}")

# 실제 사용: x_batch = x_train[batch_idx]
#            t_batch = t_train[batch_idx]


# %% [Block 4] 04-01-03-E 수치 미분 — "아주 살짝 바꿔서 변화량 측정"
def numerical_diff(f, x, h=1e-4):
    """수치 미분: 아주 작은 h로 기울기를 근사 계산.
    비유: '언덕에서 한 발짝 앞·뒤로 서보고 경사를 느끼는 것'"""
    return (f(x + h) - f(x - h)) / (2 * h)  # 중앙 차분

# ── 예제: f(x) = x² 의 미분 → 이론값은 2x ──
def f(x):
    return x ** 2

print(f"f(3) = {f(3)}")                     # 9
print(f"f'(3) ≈ {numerical_diff(f, 3):.6f}")  # 이론값: 2×3 = 6
print(f"f'(5) ≈ {numerical_diff(f, 5):.6f}")  # 이론값: 2×5 = 10


# %% [Block 5] 04-01-03-F 경사 하강법 — "내리막 방향으로 한 걸음씩"
def f(x):
    return (x - 3) ** 2                   # 최솟값은 x=3에서 0

x = 10.0                                 # 시작점: x=10 (정답에서 멀리!)
lr = 0.1                                 # 학습률: 한 걸음의 크기

print("[경사 하강법] f(x) = (x-3)² 최소화")
for step in range(20):
    grad = numerical_diff(f, x)           # 기울기 계산
    x = x - lr * grad                    # 핵심! W ← W − lr × 기울기

    if step % 5 == 0:
        print(f"  step {step:2d}: x={x:.4f}, f(x)={f(x):.6f}")

print(f"\n최종: x={x:.4f} (정답: 3.0)")


# %% [Block 6] 04-01-03-G 학습 루프 — "5단계를 반복하라!" — 신경망 학습의 5단계 (의사 코드)
# ── 신경망 학습의 5단계 (의사 코드) ──
# 이 패턴은 이 책 전체에서 변하지 않습니다!

for epoch in range(num_epochs):         # 전체 데이터를 여러 번 반복
    for batch in mini_batches:           # 미니배치 단위로

        y_pred = model(x_batch)          # ① 예측 (순전파)
        loss = cross_entropy_error(       # ② 채점 (손실 계산)
                    y_pred, t_batch)
        model.cleargrads()               # ③ 기울기 초기화
        loss.backward()                   # ④ 역전파 (기울기 계산)
        optimizer.update()                # ⑤ 파라미터 갱신 (경사 하강)

# ── 각 단계의 역할 ──
# ① 예측:      입력 → 순전파 → 출력 (04-01-02)
# ② 채점:      출력 vs 정답 → 손실 (이번 장 B,C)
# ③ 초기화:    이전 기울기 지우기 (안 하면 누적됨!)
# ④ 역전파:    손실 → 각 가중치의 기울기 (04-01-04)
# ⑤ 갱신:      W ← W − lr × 기울기 (이번 장 F)
