# -*- coding: utf-8 -*-
"""
→ SQL 인젝션 취약점을 정확히 지적하는 응답

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-08장 RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기/04-08-02 프롬프트 엔지니어링과GPT초기 설정 — 좋은 질문이 좋은 답을 만든다.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-08-02-A 1) 역할 부여 (Role Prompting)
response = llm.invoke([
    ("system", "당신은 10년차 Python 백엔드 개발자입니다. 코드 리뷰를 할 때 항상 보안 취약점을 먼저 체크합니다."),
    ("human", "이 코드를 리뷰해줘: def login(user, pw): return db.query(f\"SELECT * FROM users WHERE id='{user}' AND pw='{pw}'\")")
])
# → SQL 인젝션 취약점을 정확히 지적하는 응답


# %% [Block 2] 04-08-02-B 2) Few-shot Prompting — 예시로 패턴 학습시키기
prompt = """다음 예시처럼 리뷰를 감성 분류해줘.

리뷰: "연기가 정말 훌륭했어요" → 긍정
리뷰: "시간 낭비였다" → 부정
리뷰: "그냥 그랬다" → 중립

리뷰: "기대했던 것보단 별로였지만 OST는 좋았다" →"""

print(llm.invoke(prompt).content)
# "중립" — 예시 3개만으로 03-99 부록에서 학습시켰던 분류 작업을 즉시 수행


# %% [Block 3] 04-08-02-C 3) Chain-of-Thought (CoT) — 단계별 사고 유도
prompt = """다음 문제를 단계별로 생각한 후 최종 답을 말해줘.

문제: 한 영화관에 좌석이 200개 있다. 오늘 70%가 찼고, 그중 절반이 팝콘을 샀다.
팝콘을 산 사람은 몇 명인가?"""

print(llm.invoke(prompt).content)
# "1단계: 200 * 0.7 = 140명이 입장
#  2단계: 140 * 0.5 = 70명이 팝콘 구매
#  최종 답: 70명"
# → 단계를 거치지 않으면 곱셈을 한 번에 잘못 계산하는 경우가 줄어듦
