# -*- coding: utf-8 -*-
"""
03-04-03 정규화된 로지스틱 회귀 — (Regularized Logistic Regression) — 자신만만한 분류기에게 재갈을 물리기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-04장 - 정규화와 신경망 기초/03-04-03 정규화된 로지스틱 회귀.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

# ==================================================================
# 0. 시그모이드 (03-03-03)
# ==================================================================
def sigmoid(z):
    """
    무한한 점수를 0~1 확률로 눌러 주는 함수.

    비유:
    수도꼭지는 무한히 돌릴 수 있어도
    물탱크는 100%를 넘지 못한다.
    """
    # clip(값, -500, 500) : 범위 밖은 잘라낸다 → exp 폭발(overflow) 방지
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# ==================================================================
# 1. 정규화된 비용 — 03-03-05 코드에 '한 줄' 추가
# ==================================================================
def compute_cost_reg(X, y, theta, lam):
    """
    정규화된 로지스틱 비용.

    비유:
    발표 심사표가 두 항목이다.
      ① 내용 점수 = 예측이 얼마나 맞았나 (로그 손실)
      ② 소음 벌점 = 목소리(계수)가 얼마나 컸나
    """
    m = len(y)
    h = sigmoid(X @ theta)                    # @ 는 행렬곱

    # ⚠ h가 정확히 0이나 1이면 log(0) = -무한대 → nan
    #    아주 작은 여유(eps)를 남겨 0과 1에 '닿지 않게' 한다
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)

    # ① 내용 점수 : 로그 손실
    bce = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    # ② 소음 벌점 : θ0(절편)은 면제 → [1:] 로 첫 원소 건너뛰기 ⭐
    penalty = (lam / (2 * m)) * np.sum(theta[1:] ** 2)

    return bce + penalty

# ==================================================================
# 2. 정규화된 기울기 — 03-04-02와 '완전히 동일한 모양'
# ==================================================================
def compute_gradient_reg(X, y, theta, lam):
    """
    기울기 = 데이터 기울기 + (λ/m)·θ

    비유:
    코치가 두 명.
      데이터 코치 : "이쪽으로 뛰어!"
      체중 코치   : "살 좀 빼!" (언제나 0 방향)
    """
    m = len(y)
    h = sigmoid(X @ theta)                    # ← 선형회귀와 다른 곳은 여기 딱 한 줄!
    grad = (X.T @ (h - y)) / m                # .T = 전치. 03-03-05에서 유도한 그 식

    reg = (lam / m) * theta                   # 벌금 기울기를 일단 전체에 계산하고
    reg[0] = 0                              # ⭐ 절편 자리만 0으로 되돌린다 (최다 실수 지점)

    return grad + reg

def train_logistic_reg(X, y, lam, alpha=0.5, n_iter=20000):
    """정규화된 로지스틱 회귀를 경사하강법으로 학습한다."""
    theta = np.zeros(X.shape[1])           # 0에서 출발
    for _ in range(n_iter):
        # _ 는 '쓰지 않을 변수'라는 파이썬 관례 (반복 횟수만 필요할 때)
        grad = compute_gradient_reg(X, y, theta, lam)  # ① 전부 계산
        theta = theta - alpha * grad                    # ② 동시 갱신
    return theta

# ==================================================================
# 3. 03-03-05와 똑같은 데이터 : 종양 크기 → 악성 여부
#    크기 1~4는 양성(0), 5~8은 악성(1) → '완벽히 분리 가능'
# ==================================================================
size = np.arange(1, 9, dtype=float)      # arange(1,9) = 1,2,...,8
y    = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
X    = np.column_stack([np.ones(len(size)), size])   # 절편 열 + 특성 열 → (8,2)

# ==================================================================
# 4. λ를 바꿔가며 θ가 어떻게 달라지는지 관찰
# ==================================================================
print(f"{'λ':>6} | {'θ0':>9} {'θ1':>8} | {'J':>7} | {'경계':>7} | 정확도")
print("-" * 62)

for lam in [0, 0.01, 0.1, 1, 10]:
    th = train_logistic_reg(X, y, lam)
    h = sigmoid(X @ th)
    pred = (h >= 0.5).astype(int)              # True→1, False→0
    acc = (pred == y).mean()                    # True/False의 평균 = 맞힌 비율

    # 결정 경계 : θ0 + θ1·x = 0  →  x = -θ0/θ1  (03-03-04)
    boundary = -th[0] / th[1]

    # 성능 보고용 비용은 벌금 없이(lam=0) 계산한다 ⭐
    J_pure = compute_cost_reg(X, y, th, 0)

    print(f"{lam:>6} | {th[0]:>9.4f} {th[1]:>8.4f} | {J_pure:>7.4f} | {boundary:>7.2f} | {acc:.2f}")


# %% [Block 2] sklearn 대조 코드
from sklearn.linear_model import LogisticRegression

# penalty='l2' 가 기본값 (우리가 배운 그 벌금)
# C가 작을수록 강한 정규화 ⚠
for C in [1e6, 10, 1, 0.01]:
    model = LogisticRegression(C=C, penalty='l2', max_iter=10000)
    model.fit(size.reshape(-1, 1), y)      # reshape(-1,1) : (8,) → (8,1) 2차원으로
    # sklearn은 절편을 따로 관리한다 (intercept_ / coef_)
    b = model.intercept_[0]
    w = model.coef_[0][0]
    print(f"C={C:<8} θ0={b:8.3f}  θ1={w:7.3f}  경계={-b/w:6.3f}")


# %% [Block 3] 📝 해설
import numpy as np

def gradient_with_check(X, y, theta, lam, verbose=True):
    """
    정규화된 기울기를 계산하고, 수치미분으로 스스로 검산한다.

    비유:
    계산기로 푼 답을 손으로 한 번 더 검산하는 것.
    수식을 손으로 유도했다면 반드시 이 습관을 들이자.
    """
    # ① 해석적 기울기 (우리가 유도한 공식)
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    reg = (lam / m) * theta
    reg[0] = 0                                # 절편 면제
    analytic = grad + reg

    # ② 수치적 기울기 (정의대로 아주 작게 흔들어 보기)
    #    (f(θ+h) - f(θ-h)) / 2h  ← '중앙차분', 한쪽만 보는 것보다 정확
    numeric = np.zeros_like(theta)            # 같은 모양의 0 배열
    step = 1e-6                                # 아주 작은 간격

    for j in range(len(theta)):
        tp = theta.copy(); tp[j] += step   # copy() 필수! 안 하면 원본이 바뀐다
        tm = theta.copy(); tm[j] -= step
        numeric[j] = (compute_cost_reg(X, y, tp, lam)
                      - compute_cost_reg(X, y, tm, lam)) / (2 * step)

    # ③ 비교
    max_err = np.abs(analytic - numeric).max()
    if verbose:
        print(f"최대 오차 {max_err:.2e} →",
              "✅ 통과" if max_err < 1e-6 else "❌ 유도를 다시 확인")
        # 'A if 조건 else B' = 한 줄 if문 (조건부 표현식)

    return analytic
