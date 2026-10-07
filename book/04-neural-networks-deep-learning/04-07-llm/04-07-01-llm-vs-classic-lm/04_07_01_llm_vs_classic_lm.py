# -*- coding: utf-8 -*-
"""
API 키는 환경변수로 등록 (자세한 설정은 04-08장에서 다룹니다)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-07장 대규모 언어 모델(LLM) 이해하기 — GPT의 시대는 무엇이 다른가/04-07-01 LLM과 기존 언어 모델의 차이 — _전문가 한 명_과 _백과사전 박사_.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] API 키는 환경변수로 등록 (자세한 설정은 04-08장에서 다룹니다)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 1) 수백 개의 라벨링된 학습 데이터가 반드시 있어야 함
X_train = ["최고의 영화였다", "시간 낭비였다", "배우 연기가 훌륭했다"]
y_train = ["긍정", "부정", "긍정"]

# 2) 텍스트 → 숫자 벡터로 변환 (모델은 숫자만 이해함)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_train)

# 3) 학습(fit) — 이 과정이 없으면 모델은 아무것도 예측하지 못함
model = LogisticRegression()
model.fit(X_vec, y_train)

# 4) 새로운 문장 예측
new_review = vectorizer.transform(["연기가 별로였다"])
print(model.predict(new_review))


# %% [Block 2] 4) 새로운 문장 예측
from openai import OpenAI
client = OpenAI()  # OPENAI_API_KEY 환경변수를 자동으로 읽음

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user",
         "content": "다음 리뷰의 감성을 긍정/부정/중립 중 하나로만 답해줘: "
                    "'배우들의 연기가 좋았지만 스토리가 늘어졌다'"}
    ]
)
print(response.choices[0].message.content)
# "중립" — 03-99 부록에서 수백 개 데이터로 모델을 학습시켜야 했던 작업을, 학습 없이 즉시 수행


# %% [Block 3] "중립" — 03-99 부록에서 수백 개 데이터로 모델을 학습시켜야 했던 작업을, 학습 없이 즉시 수행
import anthropic
client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수를 자동으로 읽음

response = client.messages.create(
    model="claude-sonnet-5",   # 최신 모델명은 Anthropic 공식 문서에서 확인하세요
    max_tokens=50,
    messages=[
        {"role": "user",
         "content": "다음 리뷰의 감성을 긍정/부정/중립 중 하나로만 답해줘: "
                    "'배우들의 연기가 좋았지만 스토리가 늘어졌다'"}
    ]
)
print(response.content[0].text)


# %% [Block 4] "중립" — 03-99 부록에서 수백 개 데이터로 모델을 학습시켜야 했던 작업을, 학습 없이 즉시 수행
few_shot_prompt = """아래 예시처럼 리뷰의 감성을 분류해줘.

리뷰: "완벽한 연출이었다" → 긍정
리뷰: "다시는 안 볼 것 같다" → 부정
리뷰: "그냥저냥 볼만했다" → 중립

리뷰: "배우들의 연기가 좋았지만 스토리가 늘어졌다" → """

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": few_shot_prompt}]
)
print(response.choices[0].message.content)
