# -*- coding: utf-8 -*-
"""
kernel='linear': 직선(평면) 경계 사용 (텍스트처럼 고차원엔 보통 이걸로 충분)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-99 부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들/03-99-02 🧰 일반적인 머신러닝 모델 한눈에 — 분류기 도구함.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 03-99-02-A 1) 로지스틱 회귀 (Logistic Regression) — 로지스틱 회귀 모델 생성 및 학습
# 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression(max_iter=1000)  # max_iter: 경사하강법 반복 횟수 상한
model.fit(X_train, y_train)                # 가중치 w, 편향 b를 학습

print(model.coef_[:, :5])   # 학습된 가중치 중 앞 5개 (단어별 영향력)
print(model.intercept_)     # 편향 b


# %% [Block 2] 03-99-02-B 2) 결정 트리 (Decision Tree)
tree = DecisionTreeClassifier(
    max_depth=10,        # 트리가 자랄 수 있는 최대 깊이 (과적합 방지)
    criterion='gini',    # 분할 기준: 지니 불순도
)
tree.fit(X_train, y_train)
print(tree.get_depth())  # 실제로 자란 트리의 깊이 확인


# %% [Block 3] 03-99-02-C 3) 랜덤 포레스트 (Random Forest)
forest = RandomForestClassifier(
    n_estimators=100,   # 트리 100개를 만들어 투표
    max_features='sqrt' # 분할마다 단어 후보를 sqrt(전체 단어 수)개만 무작위로 검토
)
forest.fit(X_train, y_train)
print(forest.feature_importances_[:5])  # 각 단어(피처)가 결정에 얼마나 기여했는지


# %% [Block 4] 03-99-02-D 4) SVM (Support Vector Machine)
svm = SVC(kernel='linear', C=1.0)
# kernel='linear': 직선(평면) 경계 사용 (텍스트처럼 고차원엔 보통 이걸로 충분)
# C: 마진을 넓히는 것과 오분류를 줄이는 것 사이의 절충 강도
svm.fit(X_train, y_train)
print(svm.support_vectors_.shape)  # 지지 벡터로 선택된 데이터 개수 확인


# %% [Block 5] 03-99-02-E 5) 나이브 베이즈 (Naive Bayes)
nb = MultinomialNB(alpha=1.0)
# alpha: 라플라스 스무딩. 학습 데이터에 없던 단어가 나와도 확률이 0이 되지 않게 함
nb.fit(X_train, y_train)
print(nb.class_log_prior_)  # 클래스별 사전 확률(로그값)


# %% [Block 6] alpha: 라플라스 스무딩. 학습 데이터에 없던 단어가 나와도 확률이 0이 되지 않게 함
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 텍스트 -> TF-IDF 벡터 (5-2에서 자세히 다룸)
vec = TfidfVectorizer(max_features=5000)
Xtr = vec.fit_transform(X_train)   # 학습 데이터로 어휘 사전을 만들고 변환
Xval = vec.transform(X_val)        # 같은 어휘 사전으로 검증 데이터만 변환 (fit 하지 않음!)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=10),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear'),
    "NaiveBayes": MultinomialNB(),
}

for name, model in models.items():
    model.fit(Xtr, y_train)              # 학습
    preds = model.predict(Xval)          # 검증 데이터 예측
    print(f"{name}: {accuracy_score(y_val, preds):.3f}")
