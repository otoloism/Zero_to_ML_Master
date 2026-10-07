# -*- coding: utf-8 -*-
"""
서로 다른 알고리즘 세 개를 준비합니다 (서로 다른 "위원"들)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-99 부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들/03-99-04 🤝 앙상블 모델, 불균형 데이터, 상관 계수 — 협업과 균형.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 시각화
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 서로 다른 알고리즘 세 개를 준비합니다 (서로 다른 "위원"들)
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),   # 선형적으로 판단하는 위원
        ('rf', RandomForestClassifier(n_estimators=100)),  # 여러 갈래로 따지는 위원
        ('svm', SVC(probability=True)),              # 경계선을 예민하게 보는 위원
    ],
    voting='soft'  # 'soft' = 확률 평균, 'hard' = 다수결 투표
)

voting.fit(Xtr, y_train)          # 세 모델이 각자 독립적으로 학습
pred = voting.predict(Xval)       # 세 모델의 확률을 평균 내어 예측
print("앙상블 검증 정확도:", voting.score(Xval, y_val))


# %% [Block 2] 시각화
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 방법 1: 클래스 가중치 조정 (가장 간단, 데이터 자체는 건드리지 않음)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(Xtr, y_train)
print(classification_report(y_val, model.predict(Xval)))
#               precision  recall  f1-score
# negative          0.71     0.68     0.69    ← 소수 클래스도 합리적으로 탐지!
# positive          0.92     0.93     0.93

# 방법 2: SMOTE로 소수 클래스 합성 데이터 생성
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
print("리샘플링 후 클래스 분포:", y_resampled.value_counts())


# %% [Block 3] 시각화
import numpy as np

# 텍스트 길이와 라벨(긍정=1, 부정=0) 사이의 상관관계
label_num = (df['label'] == 'positive').astype(int)
corr = np.corrcoef(df['length'], label_num)[0, 1]   # 2x2 행렬의 비대각 원소를 꺼냄
print(f"텍스트 길이-라벨 상관계수: {corr:.3f}")
# 예: 0.05 → 거의 무관 (길이는 긍/부정 판단에 큰 의미 없음)

# 특정 단어("최고")의 등장 여부와 라벨의 상관관계
has_word = df['text'].str.contains("최고").astype(int)
corr2 = np.corrcoef(has_word, label_num)[0, 1]
print(f"'최고' 포함-라벨 상관계수: {corr2:.3f}")
# 예: 0.42 → 양의 상관 (해당 단어가 있으면 긍정일 가능성이 높음)


# %% [Block 4] 예: 0.42 → 양의 상관 (해당 단어가 있으면 긍정일 가능성이 높음)
def my_corr(x, y):
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    numerator = (x_dev * y_dev).sum()
    denominator = np.sqrt((x_dev**2).sum()) * np.sqrt((y_dev**2).sum())
    return numerator / denominator
