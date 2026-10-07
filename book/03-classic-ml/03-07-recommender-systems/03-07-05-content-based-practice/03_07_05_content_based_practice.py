# -*- coding: utf-8 -*-
"""
03-07-05 콘텐츠 기반 추천 시스템 실습 (Content Based Recommendation with TMDB 5000)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-07장 - 추천시스템/03-07-05 콘텐츠 기반 추천 시스템 실습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 1단원 — 실습 환경 세팅
import numpy as np                      # 숫자 계산
import pandas as pd                     # 표(DataFrame) 다루기

import matplotlib as mpl                # 그래프 그리기 기본 엔진
import matplotlib.pyplot as plt         # 그래프 그리는 손잡이
import seaborn as sns                   # 예쁜 그래프 스타일

import warnings                         # 경고 메시지 제어


# %% [Block 2] 1단원 — 실습 환경 세팅
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'    # 그래프를 고해상도로

mpl.rc('font', family='NanumGothic')      # 한글 폰트 설정 (없으면 한글이 □□로 깨진다)
mpl.rc('axes', unicode_minus=False)       # 마이너스 기호가 깨지지 않게

sns.set(font="NanumGothic", rc={"axes.unicode_minus": False}, style='darkgrid')
plt.rc("figure", figsize=(10, 8))         # 기본 그래프 크기

warnings.filterwarnings("ignore")         # 자잘한 경고 숨기기


# %% [Block 3] 2단원 — 데이터 불러오기와 주요 컬럼 추출 — 사용 데이터: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
# 사용 데이터: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
movies = pd.read_csv("tmdb_5000_movies.csv")   # CSV 파일을 표로 읽어들인다
movies.head(1)                                  # 맨 위 1줄만 살펴보기
movies.shape                                    # (행 개수, 열 개수)


# %% [Block 4] 2단원 — 데이터 불러오기와 주요 컬럼 추출 — 주요 컬럼으로 데이터 프레임 생성
# 주요 컬럼으로 데이터 프레임 생성
col_lst = ['id', 'title', 'genres', 'vote_average', 'vote_count',
           'popularity', 'keywords', 'overview']
movies_df = movies[col_lst]     # 대괄호 안에 컬럼 이름 리스트를 넣으면 그 열만 골라진다


# %% [Block 5] 3단원 — 전처리 : 문자열을 파이썬 객체로 되살리기
pd.set_option('max_colwidth', 80)          # 컬럼을 80자까지 넓게 출력
movies_df[['genres', 'keywords']][:1]


# %% [Block 6] 3단원 — 전처리 : 문자열을 파이썬 객체로 되살리기
from ast import literal_eval    # ast = Abstract Syntax Tree(파이썬 문법 해석기)

movies_df['genres'].apply(literal_eval)[0]


# %% [Block 7] 3단원 — 전처리 : 문자열을 파이썬 객체로 되살리기
from ast import literal_eval

# 1) 문자열을 객체로 변경: 리스트 내의 사전
movies_df['genres']   = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)

# 2) 객체에서 name만 추출: 사전마다 name 값을 꺼낸다
#    [ dic['name'] for dic in x ]  ← 리스트 컴프리헨션
#      "x 안의 사전 dic을 하나씩 꺼내서, 그 dic의 'name' 값만 모아 새 리스트를 만들어라"
movies_df['genres']   = movies_df['genres'].apply(lambda x: [dic['name'] for dic in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [dic['name'] for dic in x])

movies_df[['genres', 'keywords']][:1]


# %% [Block 8] 4단원 — CountVectorizer로 장르를 숫자로
from sklearn.feature_extraction.text import CountVectorizer

# 1) 리스트를 하나의 문자열로 변환: 공백으로 구분
#    ['Action','Adventure'] → "Action Adventure"
#    ' '.join(x) : 리스트 원소들을 공백을 사이에 두고 이어 붙인다
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x: (' ').join(x))

# 2) CountVectorizer로 단어 개수 세기
#    min_df=0     : 아무리 드물게 등장하는 단어도 버리지 않는다
#    ngram_range=(1,2) : 단어 1개짜리와 2개 연속 조합을 모두 특성으로 쓴다
#                        예) "Action", "Adventure", "Action Adventure"
count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])

print(genre_mat.shape)


# %% [Block 9] 🧪 직접 돌려보기 — 5편짜리 축소판
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 축소판 데이터 (전처리가 끝난 상태라고 가정)
movies_df = pd.DataFrame({
    "title": ["The Godfather", "GoodFellas", "Toy Story",
              "Finding Nemo", "Mad Max: Fury Road"],
    "genres": [["Drama", "Crime"], ["Drama", "Crime"],
               ["Animation", "Family", "Comedy"], ["Animation", "Family"],
               ["Action", "Adventure", "Thriller"]],
    "vote_average": [8.4, 8.2, 7.9, 7.6, 7.2],
    "vote_count":   [5893, 3128, 5546, 6292, 9427],
})

# 리스트 → 공백으로 이은 문자열
movies_df["genres_literal"] = movies_df["genres"].apply(lambda x: " ".join(x))
print(movies_df[["title", "genres_literal"]].to_string(index=False))

# 단어 세기 벡터로 변환
count_vect = CountVectorizer(min_df=1, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df["genres_literal"])
print()
print("장르 행렬 shape :", genre_mat.shape)
print("어휘 사전 크기  :", len(count_vect.vocabulary_))


# %% [Block 10] 공식
from sklearn.metrics.pairwise import cosine_similarity

# 모든 영화 쌍의 유사도를 한 번에 계산한다
genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim[0]        # 0번 영화(Avatar)와 나머지 전부의 유사도


# %% [Block 11] 공식
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print("유사도 행렬 shape :", genre_sim.shape)
print()
print("유사도 행렬 (반올림)")
print(pd.DataFrame(np.round(genre_sim, 3),
                   index=movies_df["title"], columns=movies_df["title"]).to_string())


# %% [Block 12] 6단원 — 유사 영화 추천 함수
def find_sim_movie(df, sim_matrix, title_name, top_n=10):
    """제목을 주면 장르 유사도가 높은 영화 top_n개를 돌려준다."""

    # 입력한 영화의 index 찾기
    title_movie = df[df['title'] == title_name]   # 제목이 일치하는 행만 골라낸다
    title_index = title_movie.index.values        # 그 행의 번호(인덱스)

    # 입력한 영화의 유사도를 데이터 프레임에 새 컬럼으로 추가
    #   reshape(-1, 1) : (4803,) 모양을 (4803, 1) 세로 모양으로 바꾼다
    #   -1은 "나머지는 알아서 계산해"라는 뜻
    df["similarity"] = sim_matrix[title_index, :].reshape(-1, 1)

    # 유사도 내림차순 정렬 후 상위 index 추출
    temp = df.sort_values(by="similarity", ascending=False)
    final_index = temp.index.values[: top_n]

    return df.iloc[final_index]     # iloc : 번호로 행을 골라내기

# The Godfather(대부)와 장르별 유사도가 높은 영화 10개
similar_movies = find_sim_movie(movies_df, genre_sim, 'The Godfather', 10)
similar_movies[['title', 'vote_average', "similarity"]]


# %% [Block 13] 6단원 — 유사 영화 추천 함수
def find_sim_movie(df, sim_matrix, title_name, top_n=3):
    """제목을 주면 유사도가 높은 영화 top_n개를 돌려준다(자기 자신 포함)."""
    title_index = df[df["title"] == title_name].index.values
    df = df.copy()                                  # 원본 훼손 방지
    df["similarity"] = sim_matrix[title_index, :].reshape(-1, 1)
    temp = df.sort_values(by="similarity", ascending=False)
    return temp.iloc[:top_n]

res = find_sim_movie(movies_df, genre_sim, "The Godfather", 3)
print(res[["title", "vote_average", "vote_count", "similarity"]].to_string(index=False))


# %% [Block 14] 직관 — 저울의 눈금
percentile = 0.6                                     # 상위 40% 경계를 기준으로
m = movies_df['vote_count'].quantile(percentile)     # 기준 투표 수
C = movies_df['vote_average'].mean()                 # 전체 평균 평점

def weighted_vote_average(record):
    """한 행(영화)을 받아 가중 평점을 계산한다."""
    v = record['vote_count']       # 이 영화의 투표 수
    R = record['vote_average']     # 이 영화의 평균 평점
    return ((v / (v + m)) * R) + ((m / (m + v)) * C)

# axis=1 : "행 단위로" 함수를 적용하라 (axis=0이면 열 단위)
movies_df['weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1)

temp = movies_df[['title', 'vote_average', 'vote_count', 'weighted_vote']]
temp.sort_values('weighted_vote', ascending=False)[:10]


# %% [Block 15] 직관 — 저울의 눈금
percentile = 0.6
m = movies_df["vote_count"].quantile(percentile)   # 기준 투표 수
C = movies_df["vote_average"].mean()               # 전체 평균 평점
print("m(기준 투표 수) = %.1f,  C(전체 평균 평점) = %.4f" % (m, C))

def weighted_vote_average(record):
    v = record["vote_count"]
    R = record["vote_average"]
    return ((v / (v + m)) * R) + ((m / (m + v)) * C)

movies_df["weighted_vote"] = movies_df.apply(weighted_vote_average, axis=1)
print()
print(movies_df[["title", "vote_average", "vote_count", "weighted_vote"]]
      .round(4).sort_values("weighted_vote", ascending=False).to_string(index=False))


# %% [Block 16] 🚀 확장 — 유사도와 가중 평점을 결합한 최종 추천 함수
def find_sim_movie_v2(df, sim_matrix, title_name, top_n=3):
    """유사도로 후보를 넉넉히 뽑은 뒤, 가중 평점으로 최종 순위를 매긴다."""
    title_index = df[df["title"] == title_name].index.values[0]

    # 1) 유사도 상위 (top_n * 2)편을 후보로 뽑는다 — 넉넉히 뽑아야 고를 여지가 생긴다
    #    argsort는 오름차순이므로 [::-1]로 뒤집어 내림차순으로 만든다
    sim_scores = sim_matrix[title_index]
    candidate_idx = sim_scores.argsort()[::-1][: top_n * 2]

    # 2) 자기 자신을 제외하고, 장르가 하나도 겹치지 않는 영화(유사도 0)도 버린다
    #    ← 이 줄이 없으면 "전혀 안 닮았지만 평점만 높은 영화"가 추천되어 버린다
    candidate_idx = [i for i in candidate_idx
                     if i != title_index and sim_scores[i] > 0]

    # 3) 후보들을 '가중 평점' 기준으로 다시 정렬해 상위 top_n편만 남긴다
    return df.iloc[candidate_idx].sort_values("weighted_vote",
                                              ascending=False)[:top_n]

for base in ["Toy Story", "The Godfather"]:
    res = find_sim_movie_v2(movies_df, genre_sim, base, 3)
    print("'%s' 추천 결과 (유사도 후보 → 가중 평점 정렬)" % base)
    print(res[["title", "vote_average", "vote_count", "weighted_vote"]]
          .round(3).to_string(index=False))
    print()


# %% [Block 17] 실습 — 실습 3 정답 — m을 바꿔가며 '무명 만점 영화'가 어떻게 되는지 관찰
# 실습 3 정답 — m을 바꿔가며 '무명 만점 영화'가 어떻게 되는지 관찰
# 비교용으로 투표 2표에 만점을 받은 가상의 무명 영화를 넣어본다
test = pd.DataFrame({
    "title":        ["The Godfather", "Mad Max: Fury Road", "무명 만점 영화"],
    "vote_average": [8.4,             7.2,                  10.0],
    "vote_count":   [5893,            9427,                 2],
})

print("%-20s %8s %8s %8s %8s" % ("영화", "m=0", "m=100", "m=1000", "m=6000"))
print("-" * 58)
for _, row in test.iterrows():
    line = "%-20s" % row["title"]
    for m_ in [0, 100, 1000, 6000]:
        v, R = row["vote_count"], row["vote_average"]
        w = (v / (v + m_)) * R + (m_ / (m_ + v)) * C
        line += " %8.2f" % w
    print(line)
print()
print("→ m=0이면 보정이 없어 무명 만점 영화가 10.00으로 1위를 차지한다.")
print("   m이 커질수록 무명 영화는 전체 평균 C(7.86) 쪽으로 끌려 내려가고,")
print("   투표가 많은 영화는 자기 평점을 거의 지킨다.")
