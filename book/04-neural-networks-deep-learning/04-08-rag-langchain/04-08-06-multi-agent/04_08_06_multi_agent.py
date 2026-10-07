# -*- coding: utf-8 -*-
"""
04-08-06 멀티 에이전트— 협력하는LLM팀 만들기

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-08장 RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기/04-08-06 멀티 에이전트— 협력하는LLM팀 만들기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 상세 풀이
class RoleAgent:
    """역할(시스템 지시문) 하나를 가진 에이전트"""
    def __init__(self, name, role, llm):
        self.name, self.role, self.llm = name, role, llm

    def run(self, task, context=""):
        prompt = (f"너는 {self.role}이다.\n"
                  f"[지금까지의 작업]\n{context}\n"
                  f"[이번에 할 일]\n{task}")
        return self.llm.invoke(prompt).content

researcher = RoleAgent("조사원", "주제의 핵심 사실을 5개 이내의 bullet로 정리하는 조사원", llm)
writer     = RoleAgent("작가",   "조사 내용을 초보자용 3문단 글로 쓰는 작가", llm)
reviewer   = RoleAgent("검토자", "사실 오류와 빠진 점을 지적하고, 문제가 없으면 '승인'이라고만 답하는 검토자", llm)

task  = "RAG가 LLM의 환각을 줄이는 원리를 설명하는 블로그 글"
facts = researcher.run(task)                    # ① 조사
draft = writer.run(task, context=facts)         # ② 초안

for turn in range(3):                           # ③ 검토 ↔ 수정 (최대 3회: 무한 루프 방지)
    review = reviewer.run("아래 초안을 검토해줘", context=draft)
    if review.strip().startswith("승인"):
        break
    draft = writer.run(f"검토 의견을 반영해 고쳐줘: {review}", context=draft)

print(draft)


# %% [Block 2] 📖 처음부터 끝까지 상세 풀이
agents  = {"조사원": researcher, "작가": writer, "검토자": reviewer}
history = []

for step in range(6):                           # 전체 대화 횟수 상한
    route_prompt = (f"작업: {task}\n"
                    f"최근 기록: {history[-3:]}\n"
                    f"다음에 일할 팀원을 {list(agents)} 중 하나로만 답해. "
                    f"작업이 끝났으면 '끝'이라고만 답해.")
    who = llm.invoke(route_prompt).content.strip()   # 감독자(Supervisor)의 결정
    if who not in agents:                            # '끝' 또는 잘못된 이름이면 종료
        break
    result = agents[who].run(task, context="\n".join(history))
    history.append(f"[{who}] {result}")

print(history[-1])
