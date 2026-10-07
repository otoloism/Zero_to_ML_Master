# -*- coding: utf-8 -*-
"""
03-07-02 콘텐츠 기반 추천 시스템 (Content Based Recommendation)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-07장 - 추천시스템/03-07-02 콘텐츠 기반 추천 시스템.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 직관 — 왜 곱해서 더하나
import numpy as np

# 강의 슬라이드 28쪽의 손계산을 그대로 코드로 확인한다
theta_alice = np.array([0, 5, 0])        # Alice의 취향: [기본점수, 로맨스, 액션]
x3 = np.array([1, 0.99, 0])              # Cute puppies of love의 성분표

# @ 기호는 "행렬/벡터 곱셈"을 뜻한다. 벡터끼리 쓰면 내적(dot product)이 된다.
#   즉 0*1 + 5*0.99 + 0*0 을 한 번에 계산해 준다.
pred = theta_alice @ x3
print("예측 별점 =", pred)


# %% [Block 2] 5단원 — NumPy 구현 : Alice의 취향을 학습시키기
import numpy as np

# ── 영화 5편의 성분표 X (x0=1 절편 포함) ─────────────────
#   각 행 = 영화 한 편,  각 열 = [x0(항상 1), x1(로맨스), x2(액션)]
X = np.array([
    [1, 0.90, 0.00],   # 0번: Love at last
    [1, 1.00, 0.01],   # 1번: Romance forever
    [1, 0.99, 0.00],   # 2번: Cute puppies of love  ← Alice가 안 본 영화
    [1, 0.10, 1.00],   # 3번: Nonstop car chases
    [1, 0.00, 0.90],   # 4번: Swords vs. karate
])

# ── Alice가 매긴 별점 (안 본 영화는 nan) ────────────────
y_alice = np.array([5, 5, np.nan, 0, 0])

titles = ["Love at last", "Romance forever", "Cute puppies of love",
          "Nonstop car chases", "Swords vs. karate"]
print("X.shape =", X.shape, " (영화 5편 × 특성 3개)")
print("Alice가 평가한 편수 =", int((~np.isnan(y_alice)).sum()))


# %% [Block 3] 5단원 — NumPy 구현 : Alice의 취향을 학습시키기
def train_theta(X, y, alpha=0.1, lam=0.0, iters=500):
    """한 사용자의 취향 벡터 theta를 경사하강법으로 학습한다.

    비유:
      X는 영화 성분표 묶음, y는 그 사람이 실제로 준 별점.
      우리는 '성분 → 별점' 환산표(theta)를 조금씩 고쳐가며
      실제 별점과 가장 가깝게 맞아떨어지는 환산표를 찾는다.

    인자:
      X     : (영화 수, 특성 수) 성분표 행렬
      y     : (영화 수,) 별점. 안 본 영화는 np.nan
      alpha : 학습률(보폭). 크면 빨리 가지만 발산 위험
      lam   : 정규화 계수 lambda. 0이면 정규화 없음
      iters : 반복 횟수
    반환:
      theta : (특성 수,) 학습된 취향 벡터
      hist  : 반복마다 기록한 비용 J의 리스트
    """
    # 1) 평가한 영화만 골라낸다  ← 수식의 i : r(i,j)=1 에 해당
    mask = ~np.isnan(y)          # nan이 아닌 자리 = 평가한 자리
    Xm, ym = X[mask], y[mask]    # 불리언 배열로 행을 골라내는 '불린 인덱싱'

    # 2) 취향 벡터를 0으로 초기화 (아무 취향도 없는 백지 상태에서 출발)
    theta = np.zeros(X.shape[1])
    hist = []

    for it in range(iters):
        # 3) 예측 - 실제 = 오차 벡터
        #    Xm @ theta 는 평가한 영화들의 예측 별점을 한 번에 계산한다
        err = Xm @ theta - ym

        # 4) 기울기(gradient) 계산: sum (예측-실제) * x_k
        #    Xm.T @ err 한 줄이 수식의 시그마를 전부 처리한다
        grad = Xm.T @ err

        # 5) 정규화 항 lambda * theta 를 더한다. 단 theta[0](기본요금)은 제외!
        reg = lam * theta.copy()   # copy(): 원본 theta가 바뀌지 않도록 복사본 사용
        reg[0] = 0                 # k=0 은 정규화하지 않는다는 규칙을 코드로 구현
        grad = grad + reg

        # 6) 기울기 반대 방향으로 alpha 만큼 한 걸음 이동
        theta = theta - alpha * grad

        # 7) 현재 비용 J를 기록 (잘 내려가는지 감시용)
        J = 0.5 * np.sum(err ** 2) + 0.5 * lam * np.sum(theta[1:] ** 2)
        hist.append(J)

    return theta, hist

# ── 정규화 없이 학습 (lam=0) ─────────────────────────────
theta1, hist = train_theta(X, y_alice, alpha=0.1, lam=0.0, iters=500)

print("초기 비용 J =", round(hist[0], 4))
print("100회 후 J =", round(hist[99], 6))
print("500회 후 J =", round(hist[-1], 8))
print("학습된 Alice 취향 theta =", np.round(theta1, 4))
print("  → 기본점수 %.2f, 로맨스 계수 %.2f, 액션 계수 %.2f"
      % (theta1[0], theta1[1], theta1[2]))


# %% [Block 4] 코드 설명 — 결과 읽는 법 — 학습한 취향으로 안 본 영화(2번) 예측
# ── 학습한 취향으로 안 본 영화(2번) 예측 ────────────────
pred_all = X @ theta1          # 5편 전부의 예측 별점을 한 번에 계산

print("영화별 예측 별점")
for i, t in enumerate(titles):
    actual = "  (실제 %.0f점)" % y_alice[i] if not np.isnan(y_alice[i]) else "  ← 안 본 영화!"
    print("  %-24s %6.3f %s" % (t, pred_all[i], actual))


# %% [Block 5] 코드 설명 — 결과 읽는 법 — 정규화 세기(lambda)에 따라 취향이 어떻게 변하는지 비교
# ── 정규화 세기(lambda)에 따라 취향이 어떻게 변하는지 비교 ─
print("%-8s %-30s %-12s" % ("lambda", "학습된 theta", "2번 영화 예측"))
print("-" * 56)
# 주의: lambda가 커지면 학습률 alpha도 함께 줄여야 발산하지 않는다 (대략 alpha*lambda < 2)
for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
    th, _ = train_theta(X, y_alice, alpha=0.005, lam=lam, iters=20000)
    print("%-8.1f %-30s %-12.3f" % (lam, np.round(th, 3), X[2] @ th))


# %% [Block 6] 6단원 — 사용자 4명 전체로 확장 — 사용자 4명의 별점표 (행=영화, 열=사용자). 안 본 칸은 nan
# 사용자 4명의 별점표 (행=영화, 열=사용자). 안 본 칸은 nan
Y = np.array([
    [5,      5,      0,      0     ],
    [5,      np.nan, np.nan, 0     ],
    [np.nan, 4,      0,      np.nan],
    [0,      0,      5,      4     ],
    [0,      0,      5,      np.nan],
])
users = ["Alice", "Bob", "Carol", "Dave"]

# 사용자마다 theta를 따로 학습해 세로로 쌓는다
Theta = []
for j in range(Y.shape[1]):                 # 열(=사용자) 개수만큼 반복
    th, _ = train_theta(X, Y[:, j], alpha=0.05, lam=0.1, iters=3000)
    Theta.append(th)
Theta = np.array(Theta)                     # (사용자 4명, 특성 3개)

pred = X @ Theta.T                          # (영화 5, 특성 3) @ (특성 3, 사용자 4) = (5, 4)

print("학습된 취향 행렬 Theta (행=사용자)")
print(np.round(Theta, 3))
print()
print("예측 별점 행렬 (괄호 안이 실제값, ? = 원래 비어 있던 칸)")
for i in range(5):
    row = ""
    for j in range(4):
        tag = "?" if np.isnan(Y[i, j]) else "%.0f" % Y[i, j]
        row += "  %5.2f(%s)" % (pred[i, j], tag)
    print("%-24s%s" % (titles[i], row))


# %% [Block 7] 프레임워크 관점
from sklearn.linear_model import Ridge

mask = ~np.isnan(y_alice)
# fit_intercept=True 이면 sklearn이 절편을 따로 관리하므로 x0 열은 빼고 넣는다
model = Ridge(alpha=0.1, fit_intercept=True)
model.fit(X[mask][:, 1:], y_alice[mask])     # 1: 은 "0번 열 빼고 나머지"라는 뜻

print("sklearn 계수 :", np.round(model.coef_, 4), " 절편 :", round(model.intercept_, 4))
print("2번 영화 예측:", round(model.predict(X[2:3, 1:])[0], 4))


# %% [Block 8] 실습 — 실습 4 미리 확인 — 아무것도 평가하지 않은 사용자 Eve
# 실습 4 미리 확인 — 아무것도 평가하지 않은 사용자 Eve
y_eve = np.array([np.nan] * 5)
th_eve, _ = train_theta(X, y_eve, alpha=0.05, lam=0.1, iters=1000)
print("Eve의 theta =", th_eve)
print("Eve의 예측 별점 =", X @ th_eve)
print("→ 모든 영화에 0점을 줄 것이라 예측한다. 이건 쓸모가 없다!")
