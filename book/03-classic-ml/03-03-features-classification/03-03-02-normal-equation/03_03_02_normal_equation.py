# -*- coding: utf-8 -*-
"""
03-03-02 정규 방정식 — (Computing Parameters Analytically — Normal Equation) — 산을 걸어 내려가지 않고 헬리콥터로 곧장 착륙하기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-03장 - Feature 설계와 분류 모델/03-03-02 정규방정식으로 파라미터 한 번에 계산하기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

# ------------------------------------------------------------------
# 1. 강의 자료의 집값 데이터 (m=4, 특성 4개 + 1열)
#    맨 앞 1 열은 절편 θ0 자리 (x0 = 1 트릭)
# ------------------------------------------------------------------
X = np.array([
    [1, 2104, 5, 1, 45],      # [1, 크기, 침실수, 층수, 연식]
    [1, 1416, 3, 2, 40],
    [1, 1534, 3, 2, 30],
    [1,  852, 2, 1, 36],
], dtype=float)                        # dtype=float : 정수로 두면 나눗셈에서 오차가 난다

y = np.array([460, 232, 315, 178], dtype=float)   # 집값(단위: 천 달러)

print("X shape :", X.shape, "| y shape :", y.shape)

# ------------------------------------------------------------------
# 2. 가역성 진단 — 계산 '전에' 반드시 확인하는 습관을 들이자
# ------------------------------------------------------------------
XtX = X.T @ X                                # .T = 전치(행↔열), @ = 행렬곱 → (5,4)@(4,5) = (5,5)
rank = np.linalg.matrix_rank(XtX)         # rank = '진짜로 독립적인 정보의 개수'

print("XtX 크기 :", XtX.shape, "| rank :", rank)
if rank < XtX.shape[0]:                    # rank가 크기보다 작으면 = 정보가 부족 = 비가역
    print("⚠ 비가역! inv 대신 pinv를 써야 한다.")

# ------------------------------------------------------------------
# 3. 정규 방정식 풀기 — pinv(의사역행렬)로 안전하게
#    pinv는 역행렬이 없을 때도 '가장 그럴듯한 해'를 돌려준다
# ------------------------------------------------------------------
theta = np.linalg.pinv(XtX) @ X.T @ y      # θ = (XᵀX)⁺ Xᵀ y

print("theta :", np.round(theta, 4))         # round(배열, 4) = 소수 4자리 반올림
print("예측값 :", np.round(X @ theta, 3))
print("실제값 :", y)

# ------------------------------------------------------------------
# 4. 더 안전한 실무 표준 : lstsq (최소제곱 전용 함수)
#    수치적으로 가장 안정적이며 XtX를 직접 만들지도 않는다
# ------------------------------------------------------------------
theta_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
# *_ 는 "나머지 반환값들은 관심 없으니 버린다"는 파이썬 문법(언패킹)
print("lstsq :", np.round(theta_ls, 4))


# %% [Block 2] 📝 해설
import numpy as np

def solve_normal_equation(X, y, add_bias=True):
    """
    정규 방정식으로 선형회귀 파라미터를 구한다.

    비유:
    금고를 열 때 열쇠(inv)가 맞으면 열쇠로,
    열쇠가 안 맞으면 만능 도구(pinv)로 여는 것.
    """
    X = np.asarray(X, dtype=float)     # asarray: 리스트가 와도 배열로 변환
    y = np.asarray(y, dtype=float).ravel()

    if add_bias:                        # 절편 열이 없으면 붙여 준다
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

    XtX = X.T @ X
    n_col = XtX.shape[0]                 # 정사각 행렬의 한 변 길이

    if np.linalg.matrix_rank(XtX) == n_col:   # 정보가 충분 → 가역
        inv = np.linalg.inv(XtX)
        note = "inv (가역)"
    else:                                       # 정보 부족 → 의사역행렬
        inv = np.linalg.pinv(XtX)
        note = "pinv (비가역 감지)"

    theta = inv @ X.T @ y
    return theta, note                          # 두 값을 튜플로 함께 반환
