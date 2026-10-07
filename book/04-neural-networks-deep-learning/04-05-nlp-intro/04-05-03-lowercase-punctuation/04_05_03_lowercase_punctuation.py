# -*- coding: utf-8 -*-
"""
04-05-03 소문자 변환과 특수문자·구두점 제거 — 글자 다듬기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-05장 자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음/04-05-03 소문자 변환과 특수문자·구두점 제거 — 글자 다듬기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-05-03-A 소문자 변환 — 같은 단어를 하나로 모으기
from collections import Counter

text = "Apple released a new iPhone. APPLE stock rose. apple is popular."

# 변환 전: "Apple", "APPLE", "apple"이 각각 따로 세어짐
before = Counter(text.split())
print("변환 전:", {k: v for k, v in before.items() if "pple" in k.lower()})

# 변환 후: 모두 "apple"로 통일
after = Counter(text.lower().split())
print("변환 후:", {k: v for k, v in after.items() if "apple" in k})

# ⚠️ 주의: 소문자 변환이 위험한 경우
print("\n⚠️ 'US'(미국) →", "US".lower(), "→ 'us'(우리를)와 구분 불가!")


# %% [Block 2] 04-05-03-B 특수문자·URL·이모지 제거
import re

text = "와!! 이 영화 정말 최고예요 👍👍 http://example.com #추천 #영화"
print("원본:", text)

# ① URL 제거
text = re.sub(r'http\S+|www\S+', '', text)

# ② 해시태그·멘션 제거
text = re.sub(r'[#@]\w+', '', text)

# ③ 이모지·특수문자 제거 (한글, 영문, 숫자, 공백만 남김)
#    [^\w\s가-힣] = "단어문자·공백·한글이 아닌 모든 것" = 여집합!
text = re.sub(r'[^\w\s가-힣]', ' ', text)

# ④ 중복 공백 정리
text = re.sub(r'\s+', ' ', text).strip()

print("정제 후:", text)
