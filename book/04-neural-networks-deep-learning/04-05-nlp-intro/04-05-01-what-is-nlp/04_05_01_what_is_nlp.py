# -*- coding: utf-8 -*-
"""
04-05-01 자연어 처리(NLP)란 무엇인가 — 컴퓨터에게 말을 가르치는 일

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-05장 자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음/04-05-01 자연어 처리(NLP)란 무엇인가  — 컴퓨터에게 말을 가르치는 일.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-05-01-A NLP가 왜 어려운가 — 자연어 vs 프로그래밍 언어 — "차다"라는 단어의 여러 의미
# "차다"라는 단어의 여러 의미
meanings = {
    "발로 차다":   "kick",
    "온도가 차다": "cold",
    "시간이 차다": "be full / expire",
}

for ko, en in meanings.items():
    print(f"  '{ko}' → 영어: {en}")

print("\n→ 컴퓨터는 문맥 없이는 어떤 '차다'인지 알 수 없습니다!")
print("→ 이 모호함을 해결하는 것이 NLP의 핵심 과제입니다.")


# %% [Block 2] 04-05-01-B NLP와 머신러닝의 시너지 — 텍스트 → 숫자 → 예측 — 비유: NLP = 번역기(텍스트→숫자), ML = 판단자(숫자→결론)
# 비유: NLP = 번역기(텍스트→숫자), ML = 판단자(숫자→결론)

# 1단계: NLP — 텍스트를 단어 빈도 벡터로 변환
from collections import Counter

reviews = ["이 영화 정말 최고", "지루하고 재미없다", "감동적이고 최고"]
labels  = [1, 0, 1]   # 1=긍정, 0=부정

# 간단한 "단어 존재 여부" 벡터화
vocab = sorted(set(w for r in reviews for w in r.split()))
print("어휘집합:", vocab)

for r, label in zip(reviews, labels):
    vec = [1 if w in r.split() else 0 for w in vocab]
    tag = "긍정" if label else "부정"
    print(f"  {tag} | {r:15} → 벡터: {vec}")

# 2단계: ML — 이 벡터를 분류기에 입력해 학습 (3장에서 본격 구현)
print("\n→ NLP(텍스트→벡터) + ML(벡터→분류) = 감성 분석 시스템 완성!")
