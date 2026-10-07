# -*- coding: utf-8 -*-
"""
04-05-02 텍스트 전처리 파이프라인 한눈에 보기 — 택배 분류장 비유

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-05장 자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음/04-05-02 텍스트 전처리 파이프라인 한눈에 보기 — 택배 분류장 비유.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-05-02-A 전체 파이프라인 미리보기 — 전처리 전/후 비교
import re

raw = 'Apple은 2024년에 새 iPhone을 발표했다!! 😀 (출처: news.com)'

# ① 소문자 변환 (영문에만 적용, 한글은 영향 없음)
step1 = raw.lower()
print("1단계:", step1)

# ② 특수문자/구두점/이모지/URL 제거
step2 = re.sub(r"[^\w\s가-힣]", " ", step1)
step2 = re.sub(r"\s+", " ", step2).strip()
print("2단계:", step2)

# ③ 개체명 인식 — Apple=ORG, iPhone=PRODUCT (04-05-04에서 상세)
print("3단계: [Apple→ORG, iPhone→PRODUCT, 2024→DATE]")

# ④ 품사 태깅 → 명사·동사만 추출 (04-05-05에서 상세)
tokens = ["apple", "2024", "새", "iphone", "발표"]
print("4단계:", tokens)
