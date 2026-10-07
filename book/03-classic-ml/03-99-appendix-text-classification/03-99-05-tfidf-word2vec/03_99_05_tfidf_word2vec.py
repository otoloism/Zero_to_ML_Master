# -*- coding: utf-8 -*-
"""
max_features: 상위 5000개 단어만 사용 (메모리 절약)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-99 부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들/03-99-05 🔢 TF-IDF와 Word2Vec으로 텍스트 분류하기 — 단어를 숫자로.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ⑥ 구현 관점
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# max_features: 상위 5000개 단어만 사용 (메모리 절약)
# ngram_range=(1,2): 1단어(unigram) + 2단어 묶음(bigram)까지 사용
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# fit_transform: 학습 데이터로 어휘 사전을 만들고(fit), 동시에 벡터로 변환(transform)
Xtr = vec.fit_transform(X_train)

# transform만 호출: 검증 데이터는 "이미 만든 어휘 사전"을 그대로 사용해야 함
# (검증 데이터로 새로 fit하면 안 됨 — 데이터 누수 방지)
Xval = vec.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(Xtr, y_train)
print("검증 정확도:", model.score(Xval, y_val))

# 가장 영향력 있는 단어 확인 (계수가 클수록 긍정 신호)
import numpy as np
feature_names = np.array(vec.get_feature_names_out())
top_pos = feature_names[np.argsort(model.coef_[0])[-10:]]
print("긍정 신호 단어:", top_pos)


# %% [Block 2] ⑥ 구현 관점
from gensim.models import Word2Vec
import numpy as np

# 토큰화된 문장 리스트 (앞 장의 preprocess() 함수 재사용)
sentences = [preprocess(t) for t in X_train]

# Word2Vec 모델 학습
# vector_size: 각 단어를 100차원 벡터로 표현
# window: 앞뒤 5단어까지 문맥으로 고려
# min_count: 2번 미만 등장한 단어는 무시 (노이즈 제거)
# sg=1: Skip-gram 방식 사용 (sg=0이면 CBOW 방식)
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=1)

print(w2v.wv.most_similar("최고"))
# [('훌륭하다', 0.81), ('완벽하다', 0.77), ...]

def doc_vector(tokens, model, dim=100):
    """
    문서 벡터 = 문서에 등장하는 단어 벡터들의 평균

    비유:
    반 학생들의 키를 재서 평균 키를 구하듯이,
    문서 안의 모든 단어 벡터를 더해서 평균낸 것이
    그 문서를 대표하는 하나의 벡터가 됩니다.
    """
    # model.wv에 없는 단어(학습 때 못 본 단어)는 건너뜀
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    # 단어가 하나도 없으면 0벡터 반환 (예외 처리)
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

X_train_vec = np.array([doc_vector(s, w2v) for s in sentences])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)
