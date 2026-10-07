# -*- coding: utf-8 -*-
"""
04-05-05 품사 태깅(POS Tagging) — 각 단어의 _역할_ 알아내기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-05장 자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음/04-05-05 품사 태깅(POS Tagging) — 각 단어의 _역할_ 알아내기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-05-05-A 영어 품사 태깅 (NLTK)
import nltk

text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)

print("품사 태깅 결과:")
for word, tag in tags:
    print(f"  {word:8} → {tag:5}", end="")
    if tag == "DT":  print("(한정사)")
    elif tag == "JJ": print("(형용사)")
    elif tag == "NN": print("(명사)")
    elif tag == "VBZ": print("(동사-3인칭단수)")
    elif tag == "IN": print("(전치사)")
    else: print()


# %% [Block 2] 04-05-05-B 한국어 품사 태깅 + 명사 추출 (KoNLPy)
from konlpy.tag import Okt
okt = Okt()

text = "나는 어제 친구와 맛있는 저녁을 먹었다"

# 전체 품사 태깅
pos_result = okt.pos(text)
print("품사 태깅:", pos_result)

# 어간 추출 (stem=True): "먹었다" → "먹다"
stemmed = okt.pos(text, stem=True)
print("어간 추출:", stemmed)

# 명사만 추출 → 핵심 키워드!
nouns = okt.nouns(text)
print("명사 추출:", nouns)
