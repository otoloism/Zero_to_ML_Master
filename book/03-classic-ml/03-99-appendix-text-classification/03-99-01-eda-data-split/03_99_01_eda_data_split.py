# -*- coding: utf-8 -*-
"""
positive 800

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-99 부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들/03-99-01🔍 데이터 탐색과 데이터 분할 — 요리 전 재료 점검하기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 상세 풀이
import pandas as pd

df = pd.read_csv("movie_reviews.csv")  # text, label 컬럼

print(df['label'].value_counts())
# positive    800
# negative    200
# → 4:1 불균형! (7-4에서 다룸)

df['length'] = df['text'].apply(lambda x: len(x.split()))
print(df['length'].describe())
# count, mean, std, min, 25%, 50%, 75%, max


# %% [Block 2] ④ 코드 구현
from sklearn.model_selection import train_test_split

X, y = df['text'], df['label']

# 1단계: 전체를 train(80%) / test(20%)로 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2단계: train을 다시 train(80%) / validation(20%)으로 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(len(X_train), len(X_val), len(X_test))
# 예: 640 160 200
