# -*- coding: utf-8 -*-
"""
03-06-03 이상탐지 알고리즘 — (The Anomaly Detection Algorithm) - 여러 개의 종 모양을 곱해 하나의 경보로 만들기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-06장 - 이상탐지 이론/03-06-03 이상탐지 알고리즘.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import numpy as np

# ═══ 1단계 : 특성 선택 (여기서는 열·진동 두 가지로 이미 정해짐) ═══
rng = np.random.default_rng(42)
good = rng.normal(loc=[14.0, 12.0], scale=[1.2, 1.5], size=(10000, 2))
X_train = good[:6000]        # ★ 훈련에는 정상 데이터만!

# ═══ 2단계 : 파라미터 추정 ═══════════════════════════════
def estimate_gaussian(X):
    """특성별로 μ_j 와 σ_j² 를 구한다 (QUIZ 정답 ④번 그대로)."""
    return X.mean(axis=0), X.var(axis=0)

# ═══ 3단계 : p(x) 계산 ══════════════════════════════════
def multivariate_independent_pdf(X, mu, sigma2):
    """
    독립 가정 하의 결합 확률밀도
        p(x) = ∏ⱼ p(xⱼ; μⱼ, σⱼ²)

    비유:
    자물쇠 n개를 전부 우연히 맞힐 확률을 구하는 것.
    각각의 확률을 구해서 곱하면 됩니다.

    X      : (m, n) — 판정할 데이터들
    mu     : (n,)   — 특성별 평균
    sigma2 : (n,)   — 특성별 분산
    반환   : (m,)   — 각 데이터의 p(x)
    """
    # [브로드캐스팅] X는 (m,n), mu는 (n,).
    #   NumPy가 mu 를 자동으로 m번 복사해 각 행에서 빼줍니다.
    #   → 결과는 (m, n) : "각 데이터의 각 특성별 확률"
    coef = 1.0 / np.sqrt(2 * np.pi * sigma2)        # (n,)
    exponent = -((X - mu) ** 2) / (2 * sigma2)      # (m, n)
    per_feature = coef * np.exp(exponent)              # (m, n)

    # [핵심] axis=1 로 곱하기 = "한 데이터 안에서 특성들을 전부 곱하기"
    #   np.prod 는 np.sum 의 곱셈 버전입니다.
    #   axis=0 으로 잘못 쓰면 '데이터끼리' 곱해져서 완전히 무의미해집니다!
    return np.prod(per_feature, axis=1)                # (m,)

# ── 실행 ───────────────────────────────────────────────
mu, sigma2 = estimate_gaussian(X_train)
print("mu     =", mu.round(4))
print("sigma2 =", sigma2.round(4))

# 검사할 엔진 5대 (다양한 상황을 일부러 골랐습니다)
tests = np.array([
    [14.0, 12.0],   # 평균 근처 — 완벽히 정상
    [18.5, 12.0],   # 열만 이상하게 높음
    [16.0, 14.5],   # 둘 다 살짝 높음
    [17.5, 16.5],   # 둘 다 꽤 높음
    [19.5, 6.0],    # 명백한 불량 (열↑ 진동↓)
])

epsilon = 1e-4            # 1e-4 = 0.0001. ε 고르는 법은 다음 절에서!
p_values = multivariate_independent_pdf(tests, mu, sigma2)

for x, p in zip(tests, p_values):
    # z-score : 평균에서 표준편차 몇 배만큼 떨어졌나 (해석에 유용)
    z = (x - mu) / np.sqrt(sigma2)
    verdict = "이상 🚨" if p < epsilon else "정상 ✅"
    print(f"x={x}  z=({z[0]:+.2f}, {z[1]:+.2f})  p={p:.3e}  -> {verdict}")
