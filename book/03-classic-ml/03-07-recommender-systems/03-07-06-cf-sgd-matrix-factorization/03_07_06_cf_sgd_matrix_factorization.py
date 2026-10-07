# -*- coding: utf-8 -*-
"""
03-07-06 협업 필터링 실습 — SGD 행렬 분해 (Matrix Factorization with Stochastic Gradient Descent)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-07장 - 추천시스템/03-07-06 협업 필터링 실습 — SGD 행렬 분해.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import pandas as pd


# %% [Block 1] 4단원 — 행렬 준비와 초기화
import numpy as np

# ── 원본 행렬 R(4 x 5) 생성 ─────────────────────────────
#   행 = 사용자 4명, 열 = 아이템 5개
#   np.NaN = 아직 평점을 매기지 않은 칸
R = np.array([[4,      np.nan, np.nan, 2,      np.nan],
              [np.nan, 5,      np.nan, 3,      1     ],
              [np.nan, np.nan, 3,      4,      4     ],
              [5,      2,      1,      2,      np.nan]])

print("원본 행렬 R Shape:", R.shape)
print("평가된 칸 수:", int((~np.isnan(R)).sum()), "/", R.size)


# %% [Block 2] 4단원 — 행렬 준비와 초기화 — 잠재요인 차원 K는 3으로 가정
# ── 잠재요인 차원 K는 3으로 가정 ────────────────────────
K = 3
num_users, num_items = R.shape

# ── 임의의 P(4 x 3), Q(5 x 3) 생성 ─────────────────────
np.random.seed(1)          # 난수 씨앗 고정 → 실행할 때마다 같은 결과가 나온다

# np.random.normal : 정규분포(종 모양)에서 무작위 값을 뽑는다
#   scale=1./K : 표준편차를 1/K로 작게 잡는다.
#                왜 작게? 처음부터 큰 값이면 예측이 널뛰어 학습이 불안정해진다.
#   size=(행, 열) : 만들어낼 행렬의 모양
P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))

print("원본 행렬 P Shape:", P.shape)
print("원본 행렬 Q Shape:", Q.shape)
print()
print("초기 P (사용자 잠재요인):")
print(np.round(P, 3))


# %% [Block 3] 5단원 — RMSE : 학습 상태를 보는 계기판
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, not_nan_index):
    """실제 R 행렬과 예측 행렬의 RMSE를 계산한다. Null이 아닌 원소만 이용한다."""
    error = 0

    # 예측 R 행렬 생성: P와 Q의 전치를 곱한다
    #   @ 는 행렬 곱셈 기호. P(4x3) @ Q.T(3x5) = (4x5)
    full_pred_matrix = P @ Q.T

    # Null이 아닌 실제 R 행렬과 예측 행렬만 뽑아낸다
    #   not_nan_index는 (행번호 배열, 열번호 배열) 형태라
    #   R[not_nan_index]라고 쓰면 그 위치의 값들만 1차원으로 뽑힌다
    R_not_null = R[not_nan_index]
    full_pred_matrix_not_null = full_pred_matrix[not_nan_index]

    # RMSE 계산
    mse = mean_squared_error(R_not_null, full_pred_matrix_not_null)
    rmse = np.sqrt(mse)

    return rmse

# 실제 R 행렬에서 Null이 아닌 index를 미리 구해 둔다
# np.where(조건) : 조건이 True인 위치의 (행 배열, 열 배열)을 돌려준다
not_nan_index = np.where(np.isnan(R) == False)
print("평가된 칸의 위치(행):", not_nan_index[0])
print("평가된 칸의 위치(열):", not_nan_index[1])
print()
print("학습 전 RMSE:", round(get_rmse(R, P, Q, not_nan_index), 4))


# %% [Block 4] 6단원 — SGD 학습 루프 — 반복수, 학습률, L2 규제
# ── 반복수, 학습률, L2 규제 ─────────────────────────────
steps = 1000              # 전체 데이터를 몇 바퀴 돌 것인가
learning_rate = 0.01      # eta. 한 번에 얼마나 고칠 것인가
r_lambda = 0.01           # lambda. 값이 커지는 것에 물리는 과태료

# ── SGD 기법으로 P, Q 업데이트 ─────────────────────────
for step in range(steps):

    # Null이 아닌 행 index, 열 index, 값을 한꺼번에 꺼낸다
    #   zip(a, b, c) : 세 묶음에서 하나씩 짝지어 꺼내주는 함수
    #   → u=행번호, i=열번호, r=그 칸의 실제 평점
    for u, i, r in zip(not_nan_index[0], not_nan_index[1], R[not_nan_index]):

        # 실제 값과 예측 값의 차이인 오차 값 구함
        #   P[u, :] : u번째 사용자의 잠재요인 벡터 (길이 K)
        #   Q[i, :] : i번째 아이템의 잠재요인 벡터 (길이 K)
        r_hat_ui = P[u, :] @ Q[i, :].T      # 두 벡터의 내적 = 예측 평점
        e_ui = r - r_hat_ui                 # 오차 = 실제 - 예측

        # SGD 업데이트 공식
        #   "오차 x 상대편"으로 방향을 잡고, "- lambda x 자기자신"으로 과태료를 문다
        P[u, :] = P[u, :] + learning_rate * (e_ui * Q[i, :] - r_lambda * P[u, :])
        Q[i, :] = Q[i, :] + learning_rate * (e_ui * P[u, :] - r_lambda * Q[i, :])

    # 한 바퀴가 끝날 때마다 RMSE를 재서 잘 줄어드는지 감시한다
    rmse = get_rmse(R, P, Q, not_nan_index)

    if ((step + 1) % 50) == 0:            # 50번마다 한 번씩만 출력
        print("### iteration step: ", step + 1, " rmse: ", np.round(rmse, 3))


# %% [Block 5] 7단원 — 예측 행렬 확인
pred_matrix = P @ Q.T

print('예측 행렬:')
print(np.round(pred_matrix, 3))

print("-" * 35)

print('실제 행렬:')
print(R)


# %% [Block 6] 7단원 — 예측 행렬 확인 — 빈칸이 어떻게 채워졌는지 보기 좋게 출력
# 빈칸이 어떻게 채워졌는지 보기 좋게 출력 ---------------------
print("%-10s %10s %10s" % ("위치(u,i)", "실제", "예측"))
print("-" * 34)
for u in range(R.shape[0]):
    for i in range(R.shape[1]):
        actual = "빈칸" if np.isnan(R[u, i]) else "%.0f" % R[u, i]
        mark = "  ← 새로 채워진 칸" if np.isnan(R[u, i]) else ""
        print("(%d, %d)     %10s %10.2f%s" % (u, i, actual, pred_matrix[u, i], mark))


# %% [Block 7] 8단원 — 재사용 가능한 함수로 묶기
def matrix_factorization(R, K, steps=200, learning_rate=0.01,
                         r_lambda=0.01, verbose=True):
    """평점 행렬 R을 SGD로 분해해 P, Q를 돌려준다.

    인자:
      R             : (사용자, 아이템) 평점 행렬. 빈칸은 np.nan
      K             : 잠재 요인 개수
      steps         : 전체 데이터를 몇 바퀴 돌 것인가
      learning_rate : 학습률 eta
      r_lambda      : L2 규제 계수 lambda
    반환:
      P, Q : 학습된 잠재 요인 행렬
    """
    num_users, num_items = R.shape

    # 임의의 P(사용자 x K), Q(아이템 x K) 생성
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    # 실제 R 행렬에서 Null이 아닌 index
    not_nan_index = np.where(np.isnan(R) == False)

    # SGD 기법으로 P, Q 업데이트
    for step in range(steps):
        for u, i, r in zip(not_nan_index[0], not_nan_index[1], R[not_nan_index]):
            r_hat_ui = P[u, :] @ Q[i, :].T
            e_ui = r - r_hat_ui
            P[u, :] = P[u, :] + learning_rate * (e_ui * Q[i, :] - r_lambda * P[u, :])
            Q[i, :] = Q[i, :] + learning_rate * (e_ui * P[u, :] - r_lambda * Q[i, :])

        rmse = get_rmse(R, P, Q, not_nan_index)
        if verbose and ((step + 1) % 10) == 0:
            print("### iteration step: ", step + 1, " rmse: ", np.round(rmse, 3))

    return P, Q

# ① 먼저 짧게(30바퀴)만 돌려 함수가 도는지 확인한다
P2, Q2 = matrix_factorization(R, K=3, steps=30, verbose=True)
print("30바퀴만 돌렸을 때 예측 행렬:")
print(np.round(P2 @ Q2.T, 2))
print("→ 아직 실제 평점과 한참 멀다. 덜 익은 상태.")
print()

# ② 충분히(1000바퀴) 돌려 본다. 중간 출력은 끈다
P3, Q3 = matrix_factorization(R, K=3, steps=1000, verbose=False)
print("1000바퀴 돌렸을 때 예측 행렬:")
print(np.round(P3 @ Q3.T, 2))
print("최종 RMSE:", round(get_rmse(R, P3, Q3, not_nan_index), 4))


# %% [Block 8] ① 유저-아이템 평점 행렬 만들기 — Grouplens MovieLens 데이터
# Grouplens MovieLens 데이터
movies  = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# ratings 데이터와 movies 데이터 결합
#   on="movieId" : 두 표에서 movieId가 같은 행끼리 붙인다 (SQL의 JOIN과 같다)
rating_movies = pd.merge(ratings, movies, on="movieId")

# 사용자-아이템 평점 행렬 생성
#   pivot_table("값", "행이 될 컬럼", "열이 될 컬럼")
#   → 세로로 길게 쌓인 기록을 가로세로 표로 펼친다
ratings_matrix = rating_movies.pivot_table("rating", "userId", "title")

ratings_matrix.head(3)


# %% [Block 9] ② 예측 행렬 계산 — 예측 행렬 계산
# 예측 행렬 계산
#   .values : DataFrame에서 순수 숫자 배열(NumPy)만 꺼낸다
P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=200,
                            learning_rate=0.01, r_lambda=0.01)
pred_matrix = P @ Q.T

# 다시 DataFrame으로 감싸서 사람이 읽을 수 있게 만든다
ratings_pred_matrix = pd.DataFrame(data=pred_matrix,
                                   index=ratings_matrix.index,
                                   columns=ratings_matrix.columns)
ratings_pred_matrix.head(3)


# %% [Block 10] ③ 안 본 영화 중 예측 평점 상위 $N$편 추천 — 아직 보지 않은 영화 리스트 함수
# 아직 보지 않은 영화 리스트 함수
def get_unseen_movies(ratings_matrix, userId):

    # user_rating: userId의 아이템 평점 정보 (시리즈 형태: title을 index로 가진다)
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 null인(=아직 안 본) 영화 목록
    unseen_movie_list = user_rating[user_rating.isnull()].index.tolist()

    # 모든 영화명을 list 객체로 만듦
    movies_list = ratings_matrix.columns.tolist()

    # 한줄 for + if문으로 안 본 영화 리스트 생성
    unseen_list = [movie for movie in movies_list if movie in unseen_movie_list]

    return unseen_list

# 보지 않은 영화 중 예측 평점이 높은 순서로 시리즈 반환
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

# userId 9번이 아직 보지 않은 영화 중 예측 평점 상위 10편
unseen_list = get_unseen_movies(ratings_matrix, 9)
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

recomm_movies = pd.DataFrame(data=recomm_movies.values,
                             index=recomm_movies.index,
                             columns=['pred_score'])
recomm_movies


# %% [Block 11] 🧪 직접 돌려보기 — 축소판 MovieLens
import pandas as pd

# ── 가상의 평점 기록 만들기 (긴 형태: 한 줄에 한 평점) ──────
# 사용자 12명을 '액션파'와 '로맨스파'로 나누고, 취향에 맞는 영화에 높은 평점을 준다
rng = np.random.default_rng(0)
action  = ["Mad Max", "John Wick", "Die Hard", "The Raid", "Speed"]
romance = ["Notting Hill", "Love Letter", "Before Sunrise", "La La Land", "Her"]
all_movies = action + romance

rows = []
for uid in range(1, 13):
    is_action_fan = (uid % 2 == 1)              # 홀수 번호는 액션파
    for title in all_movies:
        if rng.random() < 0.45:                 # 45% 확률로만 평점을 남긴다(희소성 재현)
            liked = (title in action) == is_action_fan
            score = rng.normal(4.5 if liked else 1.8, 0.35)
            rows.append({"userId": uid, "title": title,
                         "rating": float(np.clip(round(score * 2) / 2, 0.5, 5))})
ratings_long = pd.DataFrame(rows)
print("평점 기록 수:", len(ratings_long), "건")
print(ratings_long.head(4).to_string(index=False))

# ── 긴 표를 사용자 × 영화 행렬로 펼치기 ────────────────
ratings_matrix = ratings_long.pivot_table("rating", "userId", "title")
print()
print("평점 행렬 shape :", ratings_matrix.shape)
print("빈칸 비율       : %.1f%%" % (100 * ratings_matrix.isnull().mean().mean()))


# %% [Block 12] 🧪 직접 돌려보기 — 축소판 MovieLens — SGD 행렬 분해로 예측 행렬 만들기
# ── SGD 행렬 분해로 예측 행렬 만들기 ───────────────────
P, Q = matrix_factorization(ratings_matrix.values, K=4, steps=200,
                            learning_rate=0.01, r_lambda=0.01, verbose=False)
pred_matrix = P @ Q.T

ratings_pred_matrix = pd.DataFrame(data=pred_matrix,
                                   index=ratings_matrix.index,
                                   columns=ratings_matrix.columns)
print("예측 평점 행렬(일부, 소수 2자리)")
print(ratings_pred_matrix.iloc[:4, :5].round(2).to_string())


# %% [Block 13] 🧪 직접 돌려보기 — 축소판 MovieLens
def get_unseen_movies(ratings_matrix, userId):
    """해당 사용자가 아직 평점을 남기지 않은 영화 목록을 돌려준다."""
    user_rating = ratings_matrix.loc[userId, :]
    return user_rating[user_rating.isnull()].index.tolist()

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=3):
    """안 본 영화 중 예측 평점이 높은 순으로 top_n개를 돌려준다."""
    return pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]

for uid in [1, 2]:                                    # 1번=액션파, 2번=로맨스파
    seen = ratings_matrix.loc[uid].dropna()
    unseen_list = get_unseen_movies(ratings_matrix, uid)
    recomm = recomm_movie_by_userid(ratings_pred_matrix, uid, unseen_list, top_n=3)

    print("[userId %d] 이미 높은 평점을 준 영화: %s"
          % (uid, list(seen[seen >= 4].index)[:3]))
    print("  추천 결과")
    for title, score in recomm.items():
        print("    %-18s 예측 평점 %.2f" % (title, score))
    print()


# %% [Block 14] 실습 — 실습 1 정답 — K에 따른 최종 RMSE 비교
# 실습 1 정답 — K에 따른 최종 RMSE 비교
not_nan_index = np.where(np.isnan(R) == False)
for k in [1, 2, 3, 10]:
    Pk, Qk = matrix_factorization(R, K=k, steps=300, verbose=False)
    print("K=%2d  최종 RMSE = %.4f" % (k, get_rmse(R, Pk, Qk, not_nan_index)))
print()
print("→ K가 클수록 '아는 칸'은 잘 맞힌다. 하지만 평가된 칸이 12개뿐인데")
print("   파라미터가 (4+5)xK개나 되면 외워버리는 것에 가깝다(과적합).")


# %% [Block 15] 실습 — 실습 4 정답 — 순차 갱신 vs 진짜 동시 갱신
# 실습 4 정답 — 순차 갱신 vs 진짜 동시 갱신
def mf_simultaneous(R, K=3, steps=1000, lr=0.01, lam=0.01):
    """P와 Q를 '동시에' 갱신하는 버전 (옛 P를 따로 저장해 사용)"""
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(R.shape[0], K))
    Q = np.random.normal(scale=1./K, size=(R.shape[1], K))
    idx = np.where(np.isnan(R) == False)
    for _ in range(steps):
        for u, i, r in zip(idx[0], idx[1], R[idx]):
            e = r - P[u, :] @ Q[i, :].T
            p_old = P[u, :].copy()            # 갱신 전 값을 복사해 둔다
            P[u, :] = P[u, :] + lr * (e * Q[i, :] - lam * P[u, :])
            Q[i, :] = Q[i, :] + lr * (e * p_old  - lam * Q[i, :])   # 옛 P 사용
    return P, Q

Pa, Qa = matrix_factorization(R, K=3, steps=1000, verbose=False)   # 순차(강의 방식)
Pb, Qb = mf_simultaneous(R, K=3, steps=1000)                       # 동시

print("순차 갱신 RMSE :", round(get_rmse(R, Pa, Qa, not_nan_index), 5))
print("동시 갱신 RMSE :", round(get_rmse(R, Pb, Qb, not_nan_index), 5))
print()
print("→ 학습률이 충분히 작으면 두 방식의 차이는 무시할 수 있다.")
print("   다만 학습률이 크면 결과가 벌어지므로, 어느 방식인지 코드에서 확인하는 습관이 중요하다.")
