# -*- coding: utf-8 -*-
"""
04-04-10 WordNet 맛보기 — 단어의 의미 사전

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-10 WordNet 맛보기 — 단어의 의미 사전.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-10-A NLTK로 WordNet 탐색하기 — pip install nltk
# pip install nltk
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet', quiet=True)

# 1. 'car'의 Synset(동의어 집합) 조회
car_synsets = wn.synsets('car')
print("'car'의 Synset 목록:")
for s in car_synsets:
    print(f"  {s.name():20} → {s.definition()}")

# 2. 상위어 탐색 (개 → 포유류 → 동물 → ...)
print("\n'dog'의 상위어 경로:")
dog = wn.synset('dog.n.01')
for path in dog.hypernym_paths()[0]:
    print(f"  → {path.name()}")

# 3. 경로 유사도 비교
cat = wn.synset('cat.n.01')
car = wn.synset('car.n.01')
print(f"\n경로 유사도:")
print(f"  dog ↔ cat: {dog.path_similarity(cat):.3f} (같은 동물!)")
print(f"  dog ↔ car: {dog.path_similarity(car):.3f} (관계 먼 단어)")
