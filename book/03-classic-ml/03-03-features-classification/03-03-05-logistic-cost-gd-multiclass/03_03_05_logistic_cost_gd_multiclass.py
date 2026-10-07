# -*- coding: utf-8 -*-
"""
03-03-05 비용 함수·경사하강법·다중 클래스 분류 — (Cost Function, Gradient Descent & Multi-Class Classification) — 로그가 만드는 그릇, 그리고 셋 이상 나누기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-03장 - Feature 설계와 분류 모델/03-03-05 비용 함수·경사하강법·다중 클래스 분류.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

# ==================================================================
# 0. 도구 함수들
# ==================================================================
def sigmoid(z):
    """무한대의 값을 0~1 확률로 눌러 주는 함수 (03-03-03)."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    # clip으로 -500~500 밖의 값을 잘라 exp 폭발(overflow)을 막는다

def compute_cost(X, y, theta):
    """
    이진 교차 엔트로피(로그 손실)를 계산한다.

    비유:
    시험 채점표. 확신하며 틀린 답일수록 감점이 급격히 커진다.
    """
    m = len(y)                                  # 샘플 개수
    h = sigmoid(X @ theta)                      # 예측 확률 (m,)

    # ⚠ 매우 중요: h가 정확히 0이나 1이면 log(0) = -무한대가 되어 nan이 뜬다
    #    그래서 아주 작은 여유(eps)를 남겨 0과 1에 '닿지 않게' 한다
    eps = 1e-15                                # 1e-15 = 0.000000000000001
    h = np.clip(h, eps, 1 - eps)

    # J = -평균( y·log(h) + (1-y)·log(1-h) )
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    # np.mean = 합을 개수로 나눈 평균 (즉 1/m ∑ 를 한 번에)

def compute_gradient(X, y, theta):
    """
    4단원에서 유도한 기울기 (1/m)·Xᵀ(h−y) 를 계산한다.

    비유:
    각 특성이 오차에 얼마나 '책임'이 있는지 집계하는 성적표.
    """
    m = len(y)
    h = sigmoid(X @ theta)                      # 예측
    error = h - y                                # 예측 - 정답 (선형회귀와 완전히 동일한 형태!)
    return (X.T @ error) / m                     # .T 전치 → (n+1,m)@(m,) = (n+1,)

def train_logistic(X, y, alpha=0.5, n_iter=1000, verbose=True):
    """경사하강법으로 로지스틱 회귀를 학습한다."""
    theta = np.zeros(X.shape[1])              # 파라미터를 0에서 출발 (특성 개수만큼)
    history = []                                 # 비용 변화를 기록할 빈 리스트

    for i in range(1, n_iter + 1):            # 1부터 n_iter까지
        grad = compute_gradient(X, y, theta)   # ① 기울기를 '먼저 전부' 계산
        theta = theta - alpha * grad             # ② 그 다음 한꺼번에 갱신 (동시 갱신!)

        cost = compute_cost(X, y, theta)
        history.append(cost)                    # append = 리스트 끝에 추가

        if verbose and i in (1, 100, 1000):
            print(f"iter {i:5d} | theta = {np.round(theta,4)} | J = {cost:.6f}")
            # {i:5d} = 정수를 5칸 폭으로 오른쪽 정렬해 출력 (표처럼 보이게)

    return theta, history

# ==================================================================
# 1. 장난감 데이터 : 종양 크기 → 악성 여부
#    크기 1~4는 양성(0), 5~8은 악성(1)
# ==================================================================
size = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y    = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# 앞에 1 열을 붙여 설계 행렬로 (x0 = 1 트릭)
X = np.column_stack([np.ones(len(size)), size])
# column_stack : 1차원 배열들을 '열'로 세워 붙인다 → (8, 2)

print("X shape :", X.shape)

# ==================================================================
# 2. 학습
# ==================================================================
theta, history = train_logistic(X, y, alpha=0.5, n_iter=1000)

# ==================================================================
# 3. 결과 확인
# ==================================================================
proba = sigmoid(X @ theta)
print("\n예측 확률 :", np.round(proba, 4))
print("예측 라벨 :", (proba >= 0.5).astype(int))
print("실제 라벨 :", y.astype(int))

# 결정 경계: θ0 + θ1·x = 0  →  x = -θ0/θ1  (03-03-04에서 배운 것)
print("결정 경계 크기 :", round(-theta[0] / theta[1], 4))

# ==================================================================
# 4. 기울기 검산 (gradient checking) — 유도가 맞았는지 수치로 확인
#    수식을 손으로 유도했으면 반드시 이 검산을 하는 습관을 들이자
# ==================================================================
test_theta = np.array([-1.0, 0.5])           # 아무 값이나 하나 골라서
ana = compute_gradient(X, y, test_theta)     # 우리가 유도한 공식

num = np.zeros_like(test_theta)              # 같은 모양의 0 배열
h_step = 1e-6
for j in range(len(test_theta)):
    tp = test_theta.copy(); tp[j] += h_step   # copy()로 복사! 안 하면 원본이 바뀐다
    tm = test_theta.copy(); tm[j] -= h_step
    num[j] = (compute_cost(X, y, tp) - compute_cost(X, y, tm)) / (2 * h_step)

print("\n해석적 기울기 :", np.round(ana, 8))
print("수치적 기울기 :", np.round(num, 8))
print("최대 오차     :", np.abs(ana - num).max())


# %% [Block 2] 코드 — One-vs-All 직접 구현 + sklearn 대조
import numpy as np
from sklearn.datasets import load_iris

# ==================================================================
# 1. 붓꽃 데이터 (클래스 3개) — 꽃잎 길이·너비 두 특성만 사용
# ==================================================================
X_raw, y_multi = load_iris(return_X_y=True)
X_raw = X_raw[:, 2:]                  # [:, 2:] = 모든 행, 2번 열부터 끝까지 (꽃잎 길이/너비)

# 특성 스케일링 (경사하강법에는 필수 — 03-02-03 🔗)
mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0)
X_s = (X_raw - mu) / sd

# 편향 열 추가
X = np.column_stack([np.ones(len(X_s)), X_s])

classes = np.unique(y_multi)          # unique = 중복 없는 값 목록 → [0 1 2]
print("클래스 목록 :", classes)

# ==================================================================
# 2. One-vs-All 학습 : 클래스 개수만큼 이진 분류기를 만든다
# ==================================================================
all_theta = []                          # 분류기들의 θ를 담을 리스트

for c in classes:
    # 핵심 한 줄: 현재 클래스만 1, 나머지 전부 0 으로 라벨 재작성
    y_binary = (y_multi == c).astype(float)
    # (배열 == 값) → True/False 배열 → astype(float)로 1.0/0.0 변환

    theta_c, _ = train_logistic(X, y_binary, alpha=0.5,
                                 n_iter=3000, verbose=False)
    all_theta.append(theta_c)
    print(f"클래스 {c} 분류기 학습 완료 | theta = {np.round(theta_c,3)}")

all_theta = np.array(all_theta)       # 리스트 → (3, 3) 배열

# ==================================================================
# 3. 예측 : 세 분류기 확률 중 가장 큰 것을 고른다 (argmax)
# ==================================================================
def predict_multiclass(X, all_theta, classes):
    """
    One-vs-All 예측.

    비유:
    심사위원 3명의 점수표를 받아, 가장 높은 점수를 준
    심사위원의 장르로 판정한다.
    """
    # (m, n+1) @ (n+1, K) → (m, K) : 샘플마다 K개의 확률이 한 줄로
    probs = sigmoid(X @ all_theta.T)
    idx = probs.argmax(axis=1)          # axis=1 → '가로 방향(열끼리)' 최댓값 위치
    return classes[idx], probs           # 위치를 실제 클래스 이름으로 변환

pred, probs = predict_multiclass(X, all_theta, classes)
accuracy = (pred == y_multi).mean()    # True/False의 평균 = 맞힌 비율
print(f"\n훈련 정확도 : {accuracy:.4f}")

# 새 데이터 하나 예측해 보기 (꽃잎 길이 4.5, 너비 1.5)
new_raw = np.array([[4.5, 1.5]])
new_X = np.column_stack([[1.0], (new_raw - mu) / sd])
p_new, prob_new = predict_multiclass(new_X, all_theta, classes)
print("세 분류기 확률 :", np.round(prob_new[0], 4))
print("→ 최종 예측 클래스 :", p_new[0])
print("확률 합계 :", round(prob_new[0].sum(), 4), "← 1이 아니다!")


# %% [Block 3] 📝 해설
import numpy as np

def predict_one_vs_all(X, all_theta, classes, normalize=False):
    """
    One-vs-All 다중 클래스 예측.

    normalize=True로 두면 K개 확률의 합이 1이 되도록 나눠 준다.
    (진짜 소프트맥스는 아니지만, 비교 해석에 편리하다.)

    비유:
    심사위원 점수를 그대로 볼 수도 있고(normalize=False),
    100점 만점으로 환산해 볼 수도 있다(normalize=True).
    """
    # (m, n+1) @ (n+1, K) → (m, K)
    z = X @ all_theta.T
    probs = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    if normalize:
        # keepdims=True : 나눗셈이 행마다 제대로 퍼지도록 (m,1) 모양 유지
        total = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(total, 1e-12)   # 0으로 나누기 방지

    idx = probs.argmax(axis=1)                    # 행마다 최댓값 '위치'
    return np.asarray(classes)[idx], probs
    # 정규화를 해도 argmax 결과(예측 클래스)는 바뀌지 않는다.
    # 모든 값을 같은 수로 나누면 순서가 유지되기 때문! (03-03-04의 단조성과 같은 논리)
