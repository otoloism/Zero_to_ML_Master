# -*- coding: utf-8 -*-
"""
내부적으로: 1) "검색이 필요하다" 판단 → 2) web_search 호출 → 3) 결과를 바탕으로 답변 생성

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-08장 RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기/04-08-05 웹 검색 자동화와프롬프트 압축— 최신 정보와 비용 절감.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 상세 풀이
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "필요하면 web_search 도구를 사용해 최신 정보를 찾아 답변해줘."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "오늘 서울 날씨와 비트코인 시세를 알려줘"})
print(result["output"])
# 내부적으로: 1) "검색이 필요하다" 판단 → 2) web_search 호출 → 3) 결과를 바탕으로 답변 생성


# %% [Block 2] 내부적으로: 1) "검색이 필요하다" 판단 → 2) web_search 호출 → 3) 결과를 바탕으로 답변 생성 — 1) 대화 요약을 통한 압축
# 1) 대화 요약을 통한 압축
from langchain_core.messages import SystemMessage

def summarize_history(history, llm):
    if len(history.messages) > 10:
        old_msgs = history.messages[:-4]  # 최근 4개를 제외한 나머지
        summary_prompt = f"다음 대화를 한두 문장으로 요약해줘:\\n{old_msgs}"
        summary = llm.invoke(summary_prompt).content
        history.clear()
        history.add_message(SystemMessage(content=f"[이전 대화 요약: {summary}]"))
        # 최근 4개 메시지는 다시 추가
    return history

# 2) LLMLingua를 활용한 압축 (pip install llmlingua)
from llmlingua import PromptCompressor
compressor = PromptCompressor()

compressed = compressor.compress_prompt(
    long_context,
    instruction="다음 내용을 바탕으로 질문에 답해줘",
    question="환불 정책이 어떻게 되나요?",
    target_token=200,  # 목표 토큰 수
)
print(compressed['compressed_prompt'])
print(f"압축률: {compressed['ratio']}")
