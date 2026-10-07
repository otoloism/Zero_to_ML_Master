# -*- coding: utf-8 -*-
"""
병렬 실행: 요약과 번역을 동시에 수행 (순차 대비 시간 단축)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-08장 RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기/04-08-04 클라우드LLM활용과 고급 체인 — 배포와 응용 패턴.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 상세 풀이 — AWS Bedrock 예시 (LangChain 통합)
# AWS Bedrock 예시 (LangChain 통합)
from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
)
print(llm.invoke("클라우드 LLM의 장점을 한 문장으로 설명해줘").content)


# %% [Block 2] 📖 처음부터 끝까지 상세 풀이
from langchain_core.runnables import RunnableParallel, RunnableBranch

# 병렬 실행: 요약과 번역을 동시에 수행 (순차 대비 시간 단축)
parallel_chain = RunnableParallel(
    summary=summarize_chain,
    translation=translate_chain,
)
result = parallel_chain.invoke({"text": long_document})
print(result["summary"], result["translation"])

# 조건 분기: 질문 유형에 따라 다른 체인으로 라우팅
branch = RunnableBranch(
    (lambda x: "코드" in x["question"], code_chain),
    (lambda x: "번역" in x["question"], translate_chain),
    general_chain,  # 기본(else) 체인
)


# %% [Block 3] 조건 분기: 질문 유형에 따라 다른 체인으로 라우팅
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 참고자료를 활용해 답해줘:\\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

base_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough(),
     "history": lambda x: x.get("history", [])}
    | prompt | llm | StrOutputParser()
)

chain_with_history = RunnableWithMessageHistory(
    base_chain, get_session_history,
    input_messages_key="question", history_messages_key="history",
)
# → "환불 정책 알려줘" 다음에 "그럼 배송비는?"이라고 물어도 맥락 유지하며 문서 검색
