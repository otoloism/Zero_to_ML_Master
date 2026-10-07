# -*- coding: utf-8 -*-
"""
03-07-04 저차원 행렬분해와 평균 정규화 (Low Rank Matrix Factorization & Mean Normalization)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-07장 - 추천시스템/03-07-04 저차원 행렬분해와 평균 정규화.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 2단원 — 왜 '저차원(Low Rank)'인가
import numpy as np

# ── 저차원 근사가 얼마나 이득인지 숫자로 확인 ──────────
def compare_size(n_m, n_u, n):
    """원본 별점표를 통째로 저장할 때 vs 두 장으로 나눠 저장할 때"""
    full = n_m * n_u                 # 별점표 전체 칸 수
    lowrank = n_m * n + n_u * n      # X와 Theta의 칸 수 합
    print("  영화 %6d편 × 사용자 %6d명, 잠재요인 n=%d" % (n_m, n_u, n))
    print("     전체 표 저장     : %12s 칸" % format(full, ","))
    print("     X + Theta 저장   : %12s 칸  (%.2f%%)"
          % (format(lowrank, ","), 100 * lowrank / full))

print("[강의 예제]")
compare_size(5, 4, 2)
print("[MovieLens small 규모]")
compare_size(9724, 610, 50)
print("[대형 서비스 규모]")
compare_size(100000, 5000000, 100)


# %% [Block 2] 4단원 — 신규 사용자 Eve 문제 — Eve 문제를 직접 눈으로 확인
# Eve 문제를 직접 눈으로 확인 ---------------------------------
# (03-07-03의 협업 필터링 함수를 그대로 다시 씁니다)
def collaborative_filtering(Y_filled, R, n_features=2, alpha=0.01,
                            lam=0.1, iters=3000, seed=1):
    n_m, n_u = Y_filled.shape
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.1, size=(n_m, n_features))
    Theta = rng.normal(scale=0.1, size=(n_u, n_features))
    for _ in range(iters):
        E = (X @ Theta.T - Y_filled) * R      # 평가한 칸만 오차를 남긴다
        gX = E @ Theta + lam * X
        gT = E.T @ X + lam * Theta
        X, Theta = X - alpha * gX, Theta - alpha * gT   # 동시 갱신
    return X, Theta

# Eve(5번째 열)를 통째로 nan으로 추가한 별점표
Y5 = np.array([
    [5,      5,      0,      0,      np.nan],
    [5,      np.nan, np.nan, 0,      np.nan],
    [np.nan, 4,      0,      np.nan, np.nan],
    [0,      0,      5,      4,      np.nan],
    [0,      0,      5,      0,      np.nan],
])
titles = ["Love at last", "Romance forever", "Cute puppies of love",
          "Nonstop car chases", "Swords vs. karate"]

R5 = (~np.isnan(Y5)).astype(float)
Y5f = np.nan_to_num(Y5)

X, Theta = collaborative_filtering(Y5f, R5, iters=20000)
print("Eve의 학습된 취향 theta =", np.round(Theta[4], 6))
print("Eve에 대한 예측 별점   =", np.round((X @ Theta.T)[:, 4], 6))
print()
print("→ 전부 0. 무엇을 먼저 추천해야 할지 순위조차 매길 수 없다.")


# %% [Block 3] 강의 슬라이드 43쪽 계산 그대로 확인하기 — 각 영화의 평균 별점 mu (nan은 빼고 계산)
# 각 영화의 평균 별점 mu (nan은 빼고 계산) --------------------
# np.nanmean : nan을 무시하고 평균을 낸다. 그냥 np.mean을 쓰면 결과가 전부 nan!
# axis=1 : 가로 방향(=한 영화에 대한 모든 사용자)으로 평균
mu = np.nanmean(Y5, axis=1)

print("영화별 평균 별점 mu")
for t, m in zip(titles, mu):
    print("  %-24s %.2f" % (t, m))

# 평균을 뺀 별점표 만들기 ------------------------------------
# mu는 (5,) 모양인데 Y5는 (5,5) 모양이다.
# mu[:, None]로 (5,1) 모양으로 바꿔주면 numpy가 열 방향으로 자동 복사해서 빼준다.
#   ← 이것을 '브로드캐스팅(broadcasting)'이라고 한다.
Y_norm = Y5 - mu[:, None]

print()
print("평균을 뺀 별점표 (강의 43쪽의 오른쪽 행렬)")
print(np.round(Y_norm, 2))


# %% [Block 4] 강의 슬라이드 43쪽 계산 그대로 확인하기 — 평균을 뺀 표로 학습
# 평균을 뺀 표로 학습 -----------------------------------------
R5   = (~np.isnan(Y_norm)).astype(float)
Y5nf = np.nan_to_num(Y_norm)

X2, Theta2 = collaborative_filtering(Y5nf, R5, n_features=2,
                                     alpha=0.01, lam=0.1, iters=3000)

# 예측할 때 평균을 다시 더한다 ← 핵심!
pred = X2 @ Theta2.T + mu[:, None]

users = ["Alice", "Bob", "Carol", "Dave", "Eve"]
print("평균 정규화를 적용한 예측 별점표")
print("%-24s%s" % ("", "".join("%9s" % u for u in users)))
for i, t in enumerate(titles):
    print("%-24s%s" % (t, "".join("%9.2f" % v for v in pred[i])))
print()
print("Eve의 예측 =", np.round(pred[:, 4], 3))
print("→ 각 영화의 '평균 별점'이 그대로 예측값이 되었다. 합리적인 기본값!")


# %% [Block 5] 6단원 — 종합 구현 : 추천 파이프라인 한 벌
def recommend(Y, n_features=2, alpha=0.01, lam=0.1, iters=3000, seed=1):
    """별점표 하나를 받아 예측표와 학습된 잠재요인을 돌려주는 전체 파이프라인.

    단계:
      1) 영화별 평균 mu를 구해 빼둔다 (평균 정규화)
      2) 협업 필터링으로 X, Theta 학습
      3) 예측할 때 mu를 다시 더한다
    """
    mu = np.nanmean(Y, axis=1)              # 1) 영화별 평균
    mu = np.nan_to_num(mu)                  #    아무도 평가 안 한 영화는 0으로
    Y_norm = Y - mu[:, None]

    R = (~np.isnan(Y_norm)).astype(float)
    Y_filled = np.nan_to_num(Y_norm)

    X, Theta = collaborative_filtering(Y_filled, R, n_features,   # 2) 학습
                                       alpha, lam, iters, seed)
    pred = X @ Theta.T + mu[:, None]        # 3) 평균 되돌리기
    return pred, X, Theta, mu

def top_n_for_user(pred, Y, titles, user_idx, user_name, top_n=3):
    """아직 안 본 영화 중 예측 별점이 높은 순으로 top_n개를 출력한다."""
    unseen = np.isnan(Y[:, user_idx])       # True인 자리 = 아직 안 본 영화
    scores = pred[:, user_idx].copy()
    scores[~unseen] = -np.inf               # 이미 본 영화는 후보에서 제외
                                            #   -inf(음의 무한대)를 넣으면 정렬 시 맨 뒤로 밀린다
    order = np.argsort(-scores)[:top_n]     # argsort는 '작은 순' 이므로 부호를 뒤집어 큰 순으로
    print("[%s] 님께 추천" % user_name)
    for rank, i in enumerate(order, 1):
        if np.isinf(scores[i]):
            continue
        print("   %d위  %-24s 예측 %.2f점" % (rank, titles[i], scores[i]))

pred, X, Theta, mu = recommend(Y5)
for j, name in enumerate(["Alice", "Bob", "Carol", "Dave", "Eve"]):
    top_n_for_user(pred, Y5, titles, j, name, top_n=3)
    print()


# %% [Block 6] 6단원 — 종합 구현 : 추천 파이프라인 한 벌 — "이 영화와 비슷한 작품" 기능
# "이 영화와 비슷한 작품" 기능 --------------------------------
def similar_movies(X, titles, base_idx, top_n=3):
    """잠재 요인 공간에서 거리가 가까운 영화를 찾는다."""
    # np.linalg.norm(v) : 벡터 v의 길이. axis=1을 주면 행마다 길이를 잰다.
    dists = np.linalg.norm(X - X[base_idx], axis=1)
    order = np.argsort(dists)               # 거리가 작은 순
    print("'%s' 와(과) 비슷한 작품" % titles[base_idx])
    for i in order[1:top_n + 1]:            # [0]은 자기 자신이므로 건너뛴다
        print("   %-24s 거리 %.3f" % (titles[i], dists[i]))

similar_movies(X, titles, 0)                # Love at last 기준
print()
similar_movies(X, titles, 3)                # Nonstop car chases 기준


# %% [Block 7] 실습 — 실습 3 정답 — 아무도 평가하지 않은 신규 영화
# 실습 3 정답 — 아무도 평가하지 않은 신규 영화
Y6 = np.vstack([Y5, [np.nan] * 5])                # 맨 아래에 신규 영화 행 추가
titles6 = titles + ["<신규 개봉작>"]
mu6 = np.nanmean(Y6, axis=1)
print("신규 영화의 mu =", mu6[-1], "  ← nan! 평균을 낼 값이 없다")
print()
pred6, X6, _, _ = recommend(Y6)
print("신규 영화에 대한 예측 =", np.round(pred6[-1], 3))
print("→ 전부 0. 평균 정규화는 '신규 사용자'는 구해주지만 '신규 아이템'은 못 구한다.")
print("   이럴 때 필요한 것이 03-07-02의 콘텐츠 기반 추천이다(장르·줄거리로 추천).")
