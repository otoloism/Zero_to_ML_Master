# -*- coding: utf-8 -*-
"""
04-05-04 개체명 인식(NER) — 문장 속 _이름표_ 찾아내기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-05장 자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음/04-05-04 개체명 인식(NER) — 문장 속 _이름표_ 찾아내기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-05-04-A 영어 NER — spaCy로 직접 해보기 — pip install spacy && python -m spacy download en_core_web_sm
# pip install spacy && python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple is buying a U.K. startup for $1 billion in September 2024."
doc = nlp(text)

print("개체명 인식 결과:")
for ent in doc.ents:
    print(f"  {ent.text:18} → {ent.label_:8} ({spacy.explain(ent.label_)})")


# %% [Block 2] 04-05-04-B 한국어 NER — Transformers 파이프라인 — pip install transformers torch
# pip install transformers torch
from transformers import pipeline

# NER로 파인튜닝된 한국어 모델 로드
ner = pipeline("ner",
               model="monologg/koelectra-base-v3-naver-ner",
               aggregation_strategy="simple")

text = "삼성전자는 2024년 1월 서울에서 갤럭시 신제품을 발표했다."
result = ner(text)

print("한국어 NER 결과:")
for r in result:
    print(f"  {r['word']:12} → {r['entity_group']}")
