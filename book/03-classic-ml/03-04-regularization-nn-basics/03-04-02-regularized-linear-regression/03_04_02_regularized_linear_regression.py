# -*- coding: utf-8 -*-
"""
03-04-02 정규화된 선형회귀 — (Regularized Linear Regression) — 한 걸음 뗄 때마다 배낭에서 짐을 조금씩 덜어내는 등산객

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-04장 - 정규화와 신경망 기초/03-04-02 정규화된 선형회귀.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

np.random.seed(0)

# ==================================================================
# 1. 비용과 기울기 — θ0만 예외 처리하는 것이 이 절의 핵심
# ==================================================================
def compute_cost_reg(X, y, theta, lam):
    """
    정규화된 선형회귀 비용.

    비유:
    영수증이 두 줄로 찍힌다.
      ① 물건값 = 예측이 얼마나 틀렸나
      ② 과태료 = 계수가 얼마나 커졌나 (단, 기본급 θ0은 면제)
    """
    m = len(y)
    error = X @ theta - y                       # @ 는 행렬곱
    sse = np.sum(error ** 2)                 # ① 오차 제곱합
    penalty = lam * np.sum(theta[1:] ** 2)  # ② [1:] = 첫 원소(절편) 건너뛰기 ⭐
    return (sse + penalty) / (2 * m)

def compute_gradient_reg(X, y, theta, lam):
    """
    정규화된 기울기 = 데이터 기울기 + (λ/m)·θ

    비유:
    코치가 두 명이다.
      데이터 코치 : "이쪽으로 뛰어!"
      체중 코치   : "살 좀 빼!" (언제나 0 방향)
    두 지시를 더해서 한 걸음을 뗀다.
    """
    m = len(y)
    error = X @ theta - y
    grad = (X.T @ error) / m                    # 데이터 코치의 지시 (.T = 전치)

    # 체중 코치의 지시를 '복사본'에 만든다
    reg = (lam / m) * theta                     # 전체에 대해 일단 계산하고
    reg[0] = 0                                # ⭐ 절편 자리만 0으로 되돌린다
    # ↑ 이 한 줄을 빼먹는 것이 이 절 최다 실수

    return grad + reg

def train_gradient_descent(X, y, lam, alpha=0.3, n_iter=5000):
    """경사하강법으로 정규화된 선형회귀를 학습한다."""
    theta = np.zeros(X.shape[1])             # 파라미터를 0에서 출발
    for i in range(n_iter):
        grad = compute_gradient_reg(X, y, theta, lam)  # ① 먼저 전부 계산
        theta = theta - alpha * grad                    # ② 그 다음 한꺼번에 갱신(동시 갱신)
    return theta

# ==================================================================
# 2. 정규 방정식 버전 : θ = (XᵀX + λL)⁻¹ Xᵀy
# ==================================================================
def train_normal_equation(X, y, lam):
    """
    반복 없이 한 번에 푼다.

    비유:
    등산객이 걸어 내려가는 대신
    측량사가 지도를 펴고 최저점 좌표를 바로 찍는다.
    """
    n_col = X.shape[1]
    L = np.eye(n_col)                          # eye(n) = 대각선이 1인 단위행렬
    L[0, 0] = 0                               # ⭐ 절편 자리만 0
    # solve(A,b)는 Ax=b를 푼다 (inv를 만드는 것보다 빠르고 안정적)
    return np.linalg.solve(X.T @ X + lam * L, X.T @ y)

# ==================================================================
# 3. 데이터 준비 (03-04-01과 동일한 9차 다항 설정)
# ==================================================================
x_raw = np.linspace(-3, 3, 10)
y = 0.5 * x_raw**2 + x_raw + 2 + np.random.randn(10) * 1.5

P = np.vstack([x_raw**k for k in range(1, 10)]).T   # [x, x², ..., x⁹] → (10,9)
P = (P - P.mean(axis=0)) / P.std(axis=0)          # 스케일링 (필수!)
X = np.hstack([np.ones((10, 1)), P])                 # 절편 열 붙이기 → (10,10)

print("X shape :", X.shape, " (m=10, n+1=10 → m ≤ n+1 이므로 비가역 위험!)")

# ==================================================================
# 4. 두 방법이 같은 답을 내는지 확인 (수식 유도 검산)
# ==================================================================
LAM = 1.0
th_gd = train_gradient_descent(X, y, LAM)
th_ne = train_normal_equation(X, y, LAM)

print("\n경사하강법 θ :", np.round(th_gd[:4], 4), "...")
print("정규방정식 θ :", np.round(th_ne[:4], 4), "...")
print("최대 차이     :", np.abs(th_gd - th_ne).max())

# ==================================================================
# 5. 비가역 해결 확인 : λ=0 vs λ>0
# ==================================================================
XtX = X.T @ X
print("\n[λ=0] rank :", np.linalg.matrix_rank(XtX), "/", XtX.shape[0])
try:
    np.linalg.inv(XtX)                        # 역행렬을 시도해 본다
    print("       → 가역")
except np.linalg.LinAlgError:                # 실패하면 이쪽으로 온다
    print("       → LinAlgError: Singular matrix 💥")

L = np.eye(len(XtX)); L[0, 0] = 0
for lam in [0, 1e-8, 0.1, 1]:
    A = XtX + lam * L
    # cond = 조건수. '이 행렬이 얼마나 다루기 힘든가'를 나타내는 숫자
    # 1에 가까울수록 좋고, 1e15 이상이면 사실상 비가역으로 본다
    print(f"λ={lam:<6} rank={np.linalg.matrix_rank(A):2d}  조건수={np.linalg.cond(A):.3e}")
    # {lam:<6} = 왼쪽 정렬 6칸, {값:.3e} = 지수 표기 소수 3자리


# %% [Block 2] 📝 해설
import numpy as np

def select_lambda(X_tr, y_tr, X_val, y_val, lambdas):
    """
    검증 오차가 가장 작은 λ를 고른다.

    비유:
    옷을 여러 사이즈 입어 보고 거울(검증 셋) 앞에서
    가장 잘 맞는 것을 고르는 일.
    ⚠ 거울은 훈련에 쓰지 않은 '새 거울'이어야 한다.
    """
    best_lam, best_err = None, np.inf   # np.inf = 무한대 (첫 비교에서 무조건 지도록)
    history = []                            # 기록용 빈 리스트

    for lam in lambdas:
        # ① 훈련 셋으로만 학습
        n_col = X_tr.shape[1]
        L = np.eye(n_col); L[0, 0] = 0      # 절편 제외
        theta = np.linalg.solve(X_tr.T @ X_tr + lam * L, X_tr.T @ y_tr)

        # ② 검증 셋으로 평가 — ⭐ 정규화 항은 넣지 않는다!
        #    "성능"은 순수하게 얼마나 잘 맞히는가로만 재야 한다
        err = np.mean((X_val @ theta - y_val) ** 2)
        history.append((lam, err))

        if err < best_err:                  # 더 좋으면 챔피언 교체
            best_lam, best_err = lam, err

    return best_lam, best_err, history
