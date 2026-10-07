# -*- coding: utf-8 -*-
"""
04-04-07 RNN을 사용한 문장 생성 —seq2seq

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-07 RNN을 사용한 문장 생성 —seq2seq.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-07-A Toy 문제: 덧셈 문자열 번역 ("57+5" → "62")
import numpy as np

def make_addition_data(n_samples=1000, max_num=99):
    """'57+5 ' → ' 62' 형태의 덧셈 문제 생성
    비유: 초등학교 수학 문제집 자동 제작기!"""
    questions, answers = [], []
    for _ in range(n_samples):
        a = np.random.randint(0, max_num + 1)
        b = np.random.randint(0, max_num + 1)
        q = f"{a}+{b}".ljust(7)       # 입력: "57+5   " (패딩)
        ans = str(a + b).rjust(4)       # 출력: "  62"   (패딩)
        questions.append(q)
        answers.append(ans)
    return questions, answers

questions, answers = make_addition_data(5)
for q, a in zip(questions, answers):
    print(f"Q: '{q}' → A: '{a}'")


# %% [Block 2] 04-04-07-B Reverse 트릭의 효과 — Reverse 트릭: 입력 문자열을 뒤집는다
# Reverse 트릭: 입력 문자열을 뒤집는다
# 비유: 영어→한국어 번역 시, 어순이 반대니까 뒤에서부터 읽으면 대응이 쉬움!

original = "나는 학생 이다"
reversed_input = original[::-1]

print("원본:  ", original)
print("뒤집기:", reversed_input)
print()
print("✅ 뒤집으면 '나는'이 인코더 마지막에 처리됨")
print("   → 디코더 첫 출력 'I'와의 거리가 가까워짐!")
print("   → 기울기가 더 잘 전달 → 학습 성능 향상")
