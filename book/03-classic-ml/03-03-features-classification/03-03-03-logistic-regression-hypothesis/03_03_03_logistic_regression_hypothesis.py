# -*- coding: utf-8 -*-
"""
03-03-03 로지스틱 회귀 — 가설의 표현 — (Logistic Regression — Hypothesis Representation) — 0과 1 사이에 갇힌 S자 곡선

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-03장 - Feature 설계와 분류 모델/03-03-03 로지스틱 회귀 - 가설의 표현.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 구현 원리
import numpy as np

# ==================================================================
# ① 순진한 시그모이드 — 교과서 그대로지만 실전에서는 위험하다
# ==================================================================
def sigmoid_naive(z):
    """수식을 그대로 옮긴 버전. z가 -800쯤 되면 exp가 폭발한다."""
    return 1 / (1 + np.exp(-z))     # np.exp(x) = e의 x제곱
                                    # z=-800이면 e^800 → 컴퓨터가 표현 못 함 → inf → 경고

# ==================================================================
# ② 수치적으로 안전한 시그모이드 ⭐ 실무에서는 반드시 이 방식
# ==================================================================
def sigmoid(z):
    """
    안전한 시그모이드.

    비유:
    큰 숫자를 다룰 때 '억 단위'와 '원 단위' 계산기를 나눠 쓰는 것과 같다.
    z가 양수면 원래 식, 음수면 수학적으로 같은 다른 식을 쓴다.
        z >= 0 :  1 / (1 + e^(-z))        (e^(-z)는 0~1이라 안전)
        z <  0 :  e^(z) / (1 + e^(z))     (e^(z)가 0~1이라 안전)
    두 식은 분모·분자에 e^z를 곱한 것뿐이라 값이 완전히 같다.
    """
    z = np.asarray(z, dtype=float)      # 리스트가 들어와도 배열로 통일
    out = np.empty_like(z)                   # 같은 모양의 빈 그릇을 미리 준비

    pos = z >= 0                             # True/False로 채워진 '마스크' 배열
    neg = ~pos                                # ~ 는 True/False를 뒤집는 기호(NOT)

    # z가 0 이상인 자리만 골라 계산 (불리언 인덱싱)
    out[pos] = 1 / (1 + np.exp(-z[pos]))

    # z가 음수인 자리만 골라 다른 식으로 계산
    exp_z = np.exp(z[neg])                   # z가 음수 → e^z 는 0~1 사이 → 절대 폭발 안 함
    out[neg] = exp_z / (1 + exp_z)

    return out

# ==================================================================
# ③ 로지스틱 가설 함수와 예측
# ==================================================================
def predict_proba(X, theta):
    """각 샘플이 class 1일 확률 P(y=1|x)을 돌려준다."""
    return sigmoid(X @ theta)             # @ 는 행렬곱. (m,n+1)@(n+1,) → (m,)

def predict_label(X, theta, threshold=0.5):
    """확률을 0/1 라벨로 자른다. threshold는 상황에 따라 조절 가능."""
    return (predict_proba(X, theta) >= threshold).astype(int)
    # (배열 >= 0.5) → True/False 배열
    # .astype(int) → True를 1로, False를 0으로 변환

# ==================================================================
# ④ 성질 검증 — 배운 내용이 진짜인지 코드로 확인 (sanity check)
# ==================================================================
zs = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
print("g(z)       :", np.round(sigmoid(zs), 4))
print("g(0)       :", sigmoid(0.0))                       # 성질 2 : 0.5 여야 한다
print("g(-z)+g(z) :", np.round(sigmoid(-zs) + sigmoid(zs), 6))  # 성질 3 : 전부 1

# 미분 공식 g'(z) = g(z)(1-g(z)) 을 수치미분과 비교해 검증
h = 1e-5                                        # 1e-5 = 0.00001 (아주 작은 간격)
num_grad = (sigmoid(zs + h) - sigmoid(zs - h)) / (2 * h)   # 중앙차분
ana_grad = sigmoid(zs) * (1 - sigmoid(zs))              # 우리가 유도한 공식
print("미분 최대 오차 :", np.abs(num_grad - ana_grad).max())

# 극단값 안전성 테스트
print("g(-1000) :", sigmoid(-1000.0), "| g(1000) :", sigmoid(1000.0))


# %% [Block 2] 📝 해설
import numpy as np

def to_label(proba, threshold=0.5):
    """
    확률 배열을 0/1 라벨로 변환한다.

    비유:
    시험 점수를 받아 '합격/불합격' 도장을 찍는 것.
    합격선(threshold)은 시험마다 바꿀 수 있다.
    """
    proba = np.asarray(proba, dtype=float)
    if not (0 < threshold < 1):        # 파이썬은 0 < t < 1 처럼 연달아 비교 가능
        raise ValueError("threshold는 0과 1 사이여야 합니다")
    return (proba >= threshold).astype(int)
