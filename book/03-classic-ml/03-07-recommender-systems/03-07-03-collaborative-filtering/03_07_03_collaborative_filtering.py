# -*- coding: utf-8 -*-
"""
03-07-03 협업 필터링 (Collaborative Filtering)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-07장 - 추천시스템/03-07-03 협업 필터링.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 6단원 — NumPy 구현 : 별점표 복원하기
import numpy as np

# ── 입력은 오직 이 별점표 하나뿐! ─────────────────────
#   행 = 영화 5편,  열 = 사용자 4명,  nan = 아직 안 봄
Y = np.array([
    [5,      5,      0,      0     ],   # Love at last
    [5,      np.nan, np.nan, 0     ],   # Romance forever
    [np.nan, 4,      0,      np.nan],   # Cute puppies of love
    [0,      0,      5,      4     ],   # Nonstop car chases
    [0,      0,      5,      np.nan],   # Swords vs. karate
])
titles = ["Love at last", "Romance forever", "Cute puppies of love",
          "Nonstop car chases", "Swords vs. karate"]
users  = ["Alice", "Bob", "Carol", "Dave"]

# ── R: 평가 여부 마스크 (수식의 r(i,j)) ───────────────
R = (~np.isnan(Y)).astype(float)   # 평가했으면 1.0, 아니면 0.0

# ── Y의 nan을 0으로 바꾼 사본 (계산 편의용) ────────────
#   주의! 0으로 바꿔도 괜찮은 이유는 항상 R을 곱해서
#   그 칸의 오차를 0으로 만들어 버리기 때문이다.
Y_filled = np.nan_to_num(Y)

n_m, n_u = Y.shape
print("영화 수 n_m =", n_m, ", 사용자 수 n_u =", n_u)
print("평가된 칸 수 =", int(R.sum()), "/", n_m * n_u)


# %% [Block 2] 6단원 — NumPy 구현 : 별점표 복원하기
def collaborative_filtering(Y_filled, R, n_features=2, alpha=0.01,
                            lam=0.1, iters=2000, seed=1, verbose=True):
    """협업 필터링: 별점표만 보고 x와 theta를 동시에 학습한다.

    비유:
      성분표도 없고 취향표도 없는 상태에서 시작한다.
      두 표를 아무렇게나 채워 넣고, "지금 표로 계산한 별점"과
      "실제 별점"의 차이를 보며 두 표를 조금씩 함께 고쳐 나간다.

    인자:
      Y_filled  : (영화, 사용자) 별점표. nan은 0으로 채워둔 상태
      R         : (영화, 사용자) 평가 여부 마스크(1/0)
      n_features: 잠재 특성 개수 n. "라벨 없는 서랍 몇 칸을 줄까"
      alpha     : 학습률
      lam       : 정규화 계수
      iters     : 반복 횟수
      seed      : 난수 씨앗. 같은 값이면 항상 같은 결과가 나온다
    반환:
      X, Theta, hist
    """
    n_m, n_u = Y_filled.shape

    # ── 1) 작은 무작위 값으로 초기화 (강의 1단계) ────────
    #   왜 0이 아니라 무작위인가?
    #   전부 0으로 두면 x도 theta도 기울기가 0이 되어 영원히 안 움직인다.
    #   또 모든 칸이 똑같으면 서랍마다 다른 역할을 맡을 수 없다(대칭성 문제).
    rng = np.random.default_rng(seed)     # 재현 가능한 난수 발생기
    X = rng.normal(scale=0.1, size=(n_m, n_features))       # 영화 성분표
    Theta = rng.normal(scale=0.1, size=(n_u, n_features))   # 사용자 취향표

    hist = []
    for it in range(1, iters + 1):
        # ── 2) 예측 - 실제 = 오차. 단, 평가한 칸만 남긴다 ──
        #   X @ Theta.T  : (영화, 특성) @ (특성, 사용자) = (영화, 사용자) 예측표
        #   * R          : 안 본 칸의 오차를 0으로 지운다 ← 수식의 r(i,j)=1 조건
        E = (X @ Theta.T - Y_filled) * R

        # ── 3) 기울기 계산 ("오차 × 상대편" + 정규화) ──────
        grad_X = E @ Theta + lam * X          # x 갱신식
        grad_T = E.T @ X + lam * Theta        # theta 갱신식

        # ── 4) 동시 갱신 ─────────────────────────────────
        #   위에서 두 기울기를 '먼저 다 계산한 뒤' 여기서 함께 적용한다.
        #   순서를 섞으면 simultaneous update 규칙이 깨진다.
        X = X - alpha * grad_X
        Theta = Theta - alpha * grad_T

        # ── 5) 비용 J 기록 ───────────────────────────────
        J = 0.5 * np.sum(E ** 2) + 0.5 * lam * (np.sum(X ** 2) + np.sum(Theta ** 2))
        hist.append(J)
        if verbose and it in (1, 50, 200, 500, 1000, 2000):
            print("  반복 %5d회  J = %8.4f" % (it, J))

    return X, Theta, hist

print("학습 시작 (성분표도 취향표도 주지 않았습니다)")
X, Theta, hist = collaborative_filtering(Y_filled, R, n_features=2,
                                         alpha=0.01, lam=0.1, iters=2000)


# %% [Block 3] 6단원 — NumPy 구현 : 별점표 복원하기
pred = X @ Theta.T          # 학습된 두 표를 곱하면 완성된 별점표가 나온다

print("예측 별점표  (괄호 = 실제값, ? = 원래 비어 있던 칸)")
print("%-24s %s" % ("", "".join("%12s" % u for u in users)))
for i in range(len(titles)):
    row = ""
    for j in range(len(users)):
        tag = "?" if R[i, j] == 0 else "%.0f" % Y_filled[i, j]
        row += "%12s" % ("%.2f(%s)" % (pred[i, j], tag))
    print("%-24s %s" % (titles[i], row))


# %% [Block 4] 6단원 — NumPy 구현 : 별점표 복원하기
print("기계가 스스로 만들어낸 영화 성분표 X")
for i, t in enumerate(titles):
    print("  %-24s %s" % (t, np.round(X[i], 3)))
print()
print("기계가 스스로 만들어낸 사용자 취향표 Theta")
for j, u in enumerate(users):
    print("  %-8s %s" % (u, np.round(Theta[j], 3)))


# %% [Block 5] 6단원 — NumPy 구현 : 별점표 복원하기 — 학습된 X로 "비슷한 영화" 찾기 — 벡터 사이 거리가 가까울수록 비슷하다
# 학습된 X로 "비슷한 영화" 찾기 — 벡터 사이 거리가 가까울수록 비슷하다
base = 0                                     # 기준 영화: Love at last
print("'%s'와 비슷한 영화 순위" % titles[base])
dists = [(np.linalg.norm(X[base] - X[i]), titles[i]) for i in range(len(titles))]
for d, t in sorted(dists):                   # 거리가 작은 순으로 정렬
    print("   거리 %.3f   %s" % (d, t))


# %% [Block 6] 프레임워크 관점 — PyTorch로 같은 일을 하면 이렇게 짧아진다 (참고용)
# PyTorch로 같은 일을 하면 이렇게 짧아진다 (참고용)
import torch

Yt = torch.tensor(Y_filled, dtype=torch.float32)
Rt = torch.tensor(R, dtype=torch.float32)

# requires_grad=True : "이 텐서는 학습 대상이니 기울기를 자동으로 계산해 달라"는 표시
Xt  = torch.randn(5, 2, requires_grad=True) * 0.1
Tht = torch.randn(4, 2, requires_grad=True) * 0.1
Xt  = Xt.detach().requires_grad_()      # 곱셈 결과에도 기울기 추적을 다시 켜준다
Tht = Tht.detach().requires_grad_()

opt = torch.optim.Adam([Xt, Tht], lr=0.05, weight_decay=0.01)  # weight_decay가 정규화
for step in range(800):
    opt.zero_grad()                                  # 지난번 기울기 지우기
    loss = (((Xt @ Tht.T - Yt) * Rt) ** 2).sum() / 2  # 비용 J의 오차항
    loss.backward()                                  # 역전파로 기울기 자동 계산
    opt.step()                                       # 한 걸음 이동
print("PyTorch 최종 loss =", round(loss.item(), 4))
print("PyTorch 예측표:")


# %% [Block 7] 실습 — 실습 3 정답 — R 마스크를 빼면 어떻게 되는가
# 실습 3 정답 — R 마스크를 빼면 어떻게 되는가
X_bad, Th_bad, _ = collaborative_filtering(Y_filled, np.ones_like(R),   # R 대신 전부 1
                                           n_features=2, alpha=0.01,
                                           lam=0.1, iters=2000, verbose=False)
print("마스크를 뺀 예측표:")
print(np.round(X_bad @ Th_bad.T, 2))
print()
print("→ 원래 '?' 였던 칸이 전부 0점 쪽으로 끌려간다.")
print("   '안 봤다'를 '싫어한다'로 가르쳤기 때문이다.")
