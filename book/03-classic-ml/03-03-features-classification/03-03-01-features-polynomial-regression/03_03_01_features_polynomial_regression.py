# -*- coding: utf-8 -*-
"""
03-03-01 특성 설계와 다항 회귀 — (Features and Polynomial Regression) — 재료를 섞어 새로운 재료를 만드는 요리사

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-03장 - Feature 설계와 분류 모델/03-03-01 특성과 다항회귀.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드 — ① NumPy 직접 구현
import numpy as np                       # 수치 계산 도구 상자를 np라는 별명으로 부른다

np.random.seed(42)                          # 난수의 '출발점'을 고정 → 매번 같은 결과가 나와 재현 가능

# ------------------------------------------------------------------
# 1. 곡선 모양의 가짜 데이터 만들기
#    실제 정답 관계 : y = 0.5x² + x + 2  (+ 약간의 잡음)
# ------------------------------------------------------------------
m = 100                                     # 데이터 개수(샘플 수)
X = np.random.rand(m, 1) * 6 - 3           # rand는 0~1 균등난수 → ×6하면 0~6 → -3하면 -3~3
                                             # (m,1) 모양: 세로로 100개, 가로로 1개인 '열벡터'
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1) * 1.0
# ** 는 거듭제곱 연산자 (X**2 = X의 제곱)
# randn은 '정규분포' 난수(평균0, 표준편차1) → 현실의 측정 오차 흉내
# rand(균등) vs randn(정규) 헷갈리지 말 것! n = normal(정규)

# ------------------------------------------------------------------
# 2. 다항 특성 만들기 : [x] → [x, x²]
#    이 한 줄이 '다항 회귀'의 전부다.
# ------------------------------------------------------------------
X_poly = np.hstack([X, X**2])                # hstack = horizontal stack(옆으로 붙이기)
                                             # (100,1)과 (100,1)을 옆으로 → (100,2)
print("특성 행렬 모양 :", X_poly.shape)         # shape는 (행, 열) 크기를 알려주는 속성

# ------------------------------------------------------------------
# 3. 특성 스케일링 (4단원에서 배운 대로 '다항 생성 후'에 실시)
# ------------------------------------------------------------------
mu  = X_poly.mean(axis=0)                    # axis=0 → '세로 방향(행끼리)' 평균 = 열마다 평균 1개씩
std = X_poly.std(axis=0)                     # 열마다 표준편차(퍼진 정도)
X_scaled = (X_poly - mu) / std               # 브로드캐스팅: (100,2) - (2,) 가 자동으로 행마다 적용됨

# ------------------------------------------------------------------
# 4. 편향 항 x0 = 1 붙이기 → θ0(절편)을 행렬곱 하나로 처리하는 트릭
# ------------------------------------------------------------------
X_b = np.hstack([np.ones((m, 1)), X_scaled])   # ones((m,1)) = 1로만 채운 (100,1) 열
                                             # 결과 (100,3) : [1, x, x²]

# ------------------------------------------------------------------
# 5. 경사하강법 — 03-02장 코드를 '한 글자도' 고치지 않았다
# ------------------------------------------------------------------
theta = np.zeros((3, 1))                     # 파라미터 3개를 0에서 출발
alpha  = 0.1                                 # 학습률 = 한 걸음의 보폭
n_iter = 1000                                # 반복 횟수

for i in range(n_iter):                      # range(1000) = 0,1,2,...,999
    y_pred = X_b @ theta                     # @ 는 '행렬 곱셈' 기호 (100,3)@(3,1) → (100,1)
    error  = y_pred - y                      # 예측 - 정답 = 오차
    grad   = (2/m) * X_b.T @ error            # .T 는 전치(행↔열 뒤집기). MSE의 미분 결과
    theta -= alpha * grad                    # -= 는 'theta = theta - ...' 의 줄임말
                                             # 기울기 반대 방향으로 한 발 → 골짜기로 내려간다

print("학습된 theta (스케일된 공간) :", theta.ravel().round(4))
# ravel()은 (3,1)을 (3,)로 납작하게 펴주는 함수 → 보기 좋게 출력

# ------------------------------------------------------------------
# 6. 원래 스케일의 계수로 되돌리기 (해석을 위해)
#    θ_scaled/std 가 원래 기울기, 절편은 평균만큼 되빼준다
# ------------------------------------------------------------------
w = theta[1:].ravel() / std              # [1:] 은 '1번째부터 끝까지' 슬라이싱 (절편 제외)
b = theta[0, 0] - (theta[1:].ravel() * mu / std).sum()
print("원래 스케일 → y ≈ %.4f x² + %.4f x + %.4f" % (w[1], w[0], b))


# %% [Block 2] 코드 — ② scikit-learn 3줄 버전
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LinearRegression
from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error

# PolynomialFeatures(degree=2) = "x를 받아 [x, x²]로 늘려주는 기계"
# include_bias=False → 1 열은 LinearRegression이 알아서 넣으므로 중복 방지
poly = PolynomialFeatures(degree=2, include_bias=False)

# make_pipeline = 여러 단계를 '컨베이어 벨트'처럼 순서대로 묶는 함수
# ① 다항 확장 → ② 스케일링 → ③ 선형회귀  (4단원의 올바른 순서 그대로!)
model = make_pipeline(poly, StandardScaler(), LinearRegression())

model.fit(X, y)                              # fit = 학습(파라미터 찾기)
y_hat = model.predict(X)                     # predict = 예측

print("MSE(2차) :", round(mean_squared_error(y, y_hat), 4))

# 비교용: 그냥 직선으로 풀었을 때
lin = LinearRegression().fit(X, y)
print("MSE(1차) :", round(mean_squared_error(y, lin.predict(X)), 4))


# %% [Block 3] 📝 해설
import numpy as np

def make_design_matrix(X):
    """
    입력 X (m×1)를 받아 [1, x, x², x³, √|x|] 설계행렬(m×5)을 만든다.

    비유:
    재료 하나(x)를 받아서 다섯 가지로 손질해
    도마 위에 나란히 늘어놓는 것과 같다.
    """
    X = X.reshape(-1, 1)          # -1은 "나머지는 알아서 계산해" 라는 뜻
    m = X.shape[0]                 # 행 개수 = 샘플 수
    return np.hstack([
        np.ones((m, 1)),          # x0 = 1  (절편용)
        X,                          # x1 = x
        X**2,                       # x2 = x²
        X**3,                       # x3 = x³
        np.sqrt(np.abs(X)),        # x4 = √|x| (abs로 음수 nan 방지)
    ])
