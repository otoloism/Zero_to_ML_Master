# -*- coding: utf-8 -*-
"""
04-05-06 전처리 파이프라인 종합 — 컨베이어 벨트 끝까지 완성하기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-05장 자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음/04-05-06 전처리 파이프라인 종합 — 컨베이어 벨트 끝까지 완성하기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-05-06-A 종합 preprocess() 함수 구현
import re
from konlpy.tag import Okt

okt = Okt()

# 분석에 의미 없는 품사 목록 (조사, 어미, 구두점, 접미사)
STOPWORD_POS = {'Josa', 'Eomi', 'Punctuation', 'Suffix'}

def preprocess(raw_text: str) -> list:
    """원본 텍스트 → 깨끗한 토큰 리스트
    비유: '자동화 공장 컨베이어 벨트'"""

    # ① 소문자 변환 (영문에만 적용, 한글 영향 없음)
    text = raw_text.lower()

    # ② URL, 이메일, 특수문자/이모지 제거
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # ③ 형태소 분석 + 어간 추출 ("먹었다" → "먹다")
    text = okt.normalize(text)          # 반복 문자 정규화
    pos_tags = okt.pos(text, stem=True) # 어간 추출

    # ④ 의미 있는 품사만 선택 (1글자 제외)
    tokens = [word for word, pos in pos_tags
              if pos not in STOPWORD_POS and len(word) > 1]

    return tokens

# ── 파이프라인 실행 ──
raw = "Apple은 2024년에 새로운 iPhone을 발표했다!! 정말 멋있어요 😀 https://news.com"
result = preprocess(raw)
print("입력:", raw)
print("출력:", result)
