# -*- coding: utf-8 -*-
"""
03-05-08 PCA 적용과 붓꽃 실습 — (Applying PCA - Speedup, Misuse, and the Iris Pipeline) - 좋은 도구일수록 쓸 자리를 가려야 한다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-08 PCA 적용과 붓꽃 실습.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 파이프라인으로 안전하게
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Pipeline : 여러 단계를 하나의 모델처럼 묶어주는 도구.
#   fit()  하면 → 각 단계가 순서대로 훈련 데이터에만 fit 됩니다
#   predict() 하면 → 각 단계가 transform 만 적용됩니다
# ★ 데이터 누수를 '구조적으로' 막아주는 가장 확실한 방법입니다.
pipe = Pipeline([
    ('scaler', StandardScaler()),                 # ① 표준화
    ('pca',    PCA(n_components=0.95)),           # ② 분산 95% 유지하는 최소 k 자동 선택
    ('clf',    LogisticRegression(max_iter=1000))  # ③ 분류기
])

pipe.fit(X_train, y_train)          # 세 단계가 훈련셋에만 fit
print("테스트 정확도:", pipe.score(X_test, y_test))
print("자동 선택된 k:", pipe.named_steps['pca'].n_components_)


# %% [Block 2] 코드 — 1. 데이터 로드 및 데이터 파악 — 실습 데이터셋 라이브러리.
# 실습 데이터셋 라이브러리.
# iris - 꽃잎 데이터, 꽃 받침/길이 등을 기반으로 꽃의 종류를 예측(분류 데이터셋)
from sklearn import datasets

# PCA를 위한 라이브러리
from sklearn.decomposition import PCA

# 자료처리/시각화 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
print(dir(iris))   # dir() : 객체가 가진 속성/메서드 이름을 전부 나열

# ── feature 선택 ──────────────────────────────────────────
# [:, [0, 2]] : 모든 행 + 0번(sepal length)과 2번(petal length) 열만
#   앞 절의 [:, :2]는 '연속 구간'이었고, 여기는 '원하는 열 번호 목록'입니다.
#   이런 방식을 '팬시 인덱싱'이라고 부릅니다.
X = iris.data[:, [0, 2]]
y = iris.target

## X feature 변수 파악
print(X.shape)

feature_names = [iris.feature_names[0], iris.feature_names[2]]
print(feature_names)

df_X = pd.DataFrame(X)      # NumPy 배열 → pandas 표(DataFrame)
print(df_X.head())          # head() : 위에서 5줄만 미리보기

## Y target 변수 파악
print(y.shape)
df_Y = pd.DataFrame(y)
print(df_Y.head())
print(set(y))               # set() : 중복 제거 → 어떤 값들이 있는지 확인
print(iris.target_names)


# %% [Block 3] 코드 — 2. 기술통계량(데이터 분포) 확인 — 결측치 여부 파악
# 결측치 여부 파악
#   isnull()  : 각 칸이 비어 있으면 True
#   .sum()    : True를 1로 세어 열별 결측 개수를 반환
# ★ 결측치가 있으면 PCA는 에러를 냅니다. 반드시 먼저 확인!
print(df_X.isnull().sum())
print(df_Y.isnull().sum())

# target data부터 확인 — 클래스가 한쪽으로 치우쳤는지(불균형) 보기
df_Y[0].value_counts().plot(kind='bar')
plt.show()

# feature data 확인 — 각 특성이 어떤 모양으로 퍼져 있는지
for i in range(df_X.shape[1]):   # shape[1] = 열 개수(=2)
    sns.displot(df_X[i])            # 히스토그램 + 밀도곡선
    plt.title(feature_names[i])
    plt.show()


# %% [Block 4] 코드 — sklearn으로 PC score 구하기 — PCA 함수를 활용하여 PC를 얻어낸다.
# PCA 함수를 활용하여 PC를 얻어낸다.
pca = PCA(n_components=2)   # feature 변수 개수가 2개
pca.fit(X)                     # ★ X만 넣습니다 (y는 사용하지 않음 = 비지도)

# explained_variance_ : 각 주성분이 담고 있는 분산의 양
#                       = 03-05-07에서 배운 S 행렬의 대각 성분(고유값)
print(pca.explained_variance_)   # 이것은 eigen value를 의미함

# transform : 학습된 축으로 데이터를 투영 → PC score (새 좌표)
PCscore = pca.transform(X)
print(PCscore[0:5])   # X의 자료에 eigen vector를 곱한 값,
                        # 새로운 공간에서 좌표값으로 나타남


# %% [Block 5] 코드 — 직접 계산으로 검산하기 — PC score 구하기
# PC score 구하기
# components_ : 주성분 벡터들이 '행'으로 저장되어 있습니다 (k, n)
# transpose() 하면 '열'이 eigen vector가 됩니다 (n, k)
#   → 03-05-07의 U_reduce 와 정확히 같은 형태!
eigens_v = pca.components_.transpose()
# column vector가 eigen vector가 되도록 transpose()
# 현재 2X2라 티가 안나지만, nxp의 경우는 shape를 통해 확인가능

# PCscore를 구하기위해 eigen vector를 곱하기 전에 centering 작업
# ★ 이 단계를 빼먹으면 값이 전혀 다르게 나옵니다 (03-05-07 2단원)
mX = X - X.mean(axis=0)     # 열별 평균을 빼서 중심화
dfmX = pd.DataFrame(mX)

# PC score를 구하기 위해 eigen vector를 곱함
print((mX @ eigens_v)[0:5])   # @ = 행렬 곱 (150,2)@(2,2) = (150,2)


# %% [Block 6] 시각화 ③ — 주성분 방향 그려보기 — PC score scatter
# PC score scatter
plt.scatter(PCscore[:, 0], PCscore[:, 1])  # 0-X축 / 1-Y축
plt.show()

# centering된 원래 데이터 위에 주성분 화살표 그리기
plt.scatter(dfmX[0], dfmX[1])
origin = [0], [0]      # origin point — 화살표 시작점(원점)

# quiver : 화살표(벡터)를 그리는 함수
#   *origin : 리스트 두 개를 x좌표들, y좌표들로 펼쳐 전달 (가변 인자 언패킹)
#   scale   : 값이 클수록 화살표가 짧아집니다
plt.quiver(*origin, eigens_v[1, :], color=["r", "b"], scale=3)
plt.show()


# %% [Block 7] 코드 — 4개 특성 전부로 PCA
X2 = iris.data                    # 이번엔 4개 특성 모두
pca2 = PCA(n_components=4)        # 4개 변수를 모두 사용
pca2.fit(X2)

print(pca2.explained_variance_)   # 각 PC의 eigen value
print(pca2.explained_variance_ratio_.cumsum())   # 누적 설명 비율

# PCs = 위에서 판단한 높은 PC 두개를 선택함
#   [:, 0:2] → 앞의 2개 열(PC1, PC2)만 사용
PCs = pca2.transform(X2)[:, 0:2]


# %% [Block 8] 코드 — 로지스틱 회귀와 혼동 행렬 — ✅ 최신 환경 수정본
# ✅ 최신 환경 수정본
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ── ① PC 2개로 학습 ──────────────────────────────────────
clf2 = LogisticRegression(
    solver="sag",        # Stochastic Average Gradient — 큰 데이터에 유리한 최적화기
    max_iter=1000,      # ★ 수렴 경고를 없애려면 충분히 크게
    random_state=0      # sag는 확률적이므로 시드 고정
).fit(PCs, y)                # ← 입력이 PC score 2개

# confusion_matrix(실제, 예측) : (i,j)칸 = 실제 i인데 j로 예측한 개수
#   → 대각선이 정답, 나머지는 오분류
print("[PC 2개]")
print(confusion_matrix(y, clf2.predict(PCs)))
print("정확도: %.4f" % accuracy_score(y, clf2.predict(PCs)))

# ── ② 원본 특성 2개(sepal length, sepal width)로 학습 ─────
clf = LogisticRegression(
    solver="sag", max_iter=1000, random_state=0
).fit(X2[:, 0:2], y)      # ← 입력이 원본 특성 2개

print("\n[원본 특성 2개]")
print(confusion_matrix(y, clf.predict(X2[:, 0:2])))
print("정확도: %.4f" % accuracy_score(y, clf.predict(X2[:, 0:2])))
