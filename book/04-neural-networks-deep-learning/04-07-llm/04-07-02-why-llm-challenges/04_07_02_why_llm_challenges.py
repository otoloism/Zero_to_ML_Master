# -*- coding: utf-8 -*-
"""
우리는 GPU도, 학습 데이터도, 파라미터 튜닝도 필요 없습니다.

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-07장 대규모 언어 모델(LLM) 이해하기 — GPT의 시대는 무엇이 다른가/04-07-02 LLM을 개발·활용하는 이유와 개발 도전 과제 — 거대한 공장을 짓는 이유.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 🟢 1단계 — 대규모 언어 모델을 "개발"하고 "활용"하는 이유 — "개발"이 아닌 "활용" — 이미 지어진 공장(모델)에 주문만 넣는 코드
# "개발"이 아닌 "활용" — 이미 지어진 공장(모델)에 주문만 넣는 코드
# 우리는 GPU도, 학습 데이터도, 파라미터 튜닝도 필요 없습니다.

def ask_llm(prompt: str) -> str:
    """
    이미 학습이 끝난 LLM(완성된 공장)에게
    "주문서(prompt)"를 보내고 "완제품(응답)"을 받는 함수.
    """
    response = call_api(
        model="claude-sonnet",   # 어떤 공장에서 만든 제품을 쓸지 선택
        messages=[{"role": "user", "content": prompt}]
    )
    return response.text  # 완제품(생성된 텍스트) 반환

# 사용 예시: 고객 문의 자동 분류
answer = ask_llm("이 문의를 '환불/배송/기타' 중 하나로 분류해줘: 배송이 너무 늦어요")


# %% [Block 2] 🔴 2단계 — 대규모 언어 모델 개발 도전 과제 — Scaling Law 개념 시각화 (실제 학습이 아닌 개념 그래프)
# Scaling Law 개념 시각화 (실제 학습이 아닌 개념 그래프)
import numpy as np
import matplotlib.pyplot as plt

# 파라미터 수: 1억 ~ 1조 (10의 거듭제곱 단위로 증가)
params = np.array([1e8, 1e9, 1e10, 1e11, 1e12])

# 멱법칙 근사: L(N) = (Nc/N)^alpha_N 형태를 단순화한 식
# 계수 2.5, 지수 0.05는 실제 논문 값이 아닌 "개념 이해용" 값
loss = 2.5 * params ** (-0.05)

plt.plot(params, loss, marker='o', color='#1d4ed8')
plt.xscale('log')  # x축을 로그 스케일로 → 거듭제곱 관계가 직선처럼 보임
plt.xlabel('파라미터 수 (log scale)')
plt.ylabel('테스트 손실(Loss)')
plt.title('Scaling Law 개념: 모델이 커질수록 손실은 예측 가능하게 감소')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()
