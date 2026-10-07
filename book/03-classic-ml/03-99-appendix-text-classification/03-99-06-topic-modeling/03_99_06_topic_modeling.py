# -*- coding: utf-8 -*-
"""
토픽 모델링은 보통 TF-IDF보다 단순 빈도(CountVectorizer)를 사용

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-99 부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들/03-99-06 🗂 토픽 모델링과 주피터 종합 실습 — 라벨 없이 주제 찾기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import pandas as pd


# %% [Block 1] 3단계 — 코드로 직접 토픽 추출하기
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 토픽 모델링은 보통 TF-IDF보다 단순 빈도(CountVectorizer)를 사용
# → LDA는 "단어가 몇 번 등장했는가"라는 카운트 자체를 확률 모델의 기반으로 삼기 때문
vec = CountVectorizer(
    max_features=1000,  # 가장 빈도 높은 상위 1000개 단어만 사용
    max_df=0.95,         # 문서의 95% 이상에 등장하는 흔한 단어는 제외
    min_df=2             # 최소 2개 문서에는 등장해야 포함 (희귀 오탈자 제거)
)
X_counts = vec.fit_transform(df['text'])

# n_components=5 → 토픽 개수 K를 5개로 가정 (사람이 미리 정하는 하이퍼파라미터)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_counts)  # 여기서 π(문서-토픽), φ(토픽-단어)를 동시에 근사 추정

# 각 토픽의 대표 단어 출력
feature_names = vec.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    # argsort()는 오름차순 정렬 인덱스를 반환하므로, 뒤에서 10개가 확률이 가장 높은 단어
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"토픽 {idx}: {', '.join(top_words)}")


# %% [Block 2] 4단계 — 주피터 노트북 종합 실습: 03-99 부록 전체를 잇는 엔드투엔드 파이프라인 — 03-99 부록 종합 파이프라인
# ── 03-99 부록 종합 파이프라인 ──

# 1. 데이터 로드 + EDA (7-1)
df = pd.read_csv("reviews.csv")
print(df['label'].value_counts())  # 클래스 비율을 먼저 눈으로 확인

# 2. 04-05장 전처리 파이프라인 적용
df['tokens'] = df['text'].apply(preprocess)
df['clean_text'] = df['tokens'].apply(lambda t: ' '.join(t))

# 3. 데이터 분할 (7-1) — stratify로 클래스 비율을 train/test에 동일하게 유지
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2,
    stratify=df['label'], random_state=42
)

# 4. TF-IDF 벡터화 (7-5) — train으로 fit, test는 transform만
vec = TfidfVectorizer(max_features=5000)
Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

# 5. 클래스 불균형 대응 (7-4)
model = LogisticRegression(class_weight='balanced', max_iter=1000)

# 6. 하이퍼파라미터 튜닝 (7-3) — 교차검증으로 C값 탐색
grid = GridSearchCV(model, {'C': [0.1, 1, 10]}, cv=5, scoring='f1')
grid.fit(Xtr, y_train)

# 7. 최종 테스트셋 평가 (단 한 번!) — 튜닝 과정에서 절대 들여다보지 않은 데이터
print(classification_report(y_test, grid.best_estimator_.predict(Xte)))
