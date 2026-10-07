# -*- coding: utf-8 -*-
"""
1) 문서 로드

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-08장 RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기/04-08-03 RAG와랭체인입문 —LLM에게 _참고자료_를 쥐어주기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 상세 풀이
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1) 문서 로드
loader = TextLoader("company_faq.txt", encoding="utf-8")
docs = loader.load()

# 2) 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3) 임베딩 + 벡터 저장
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 청크 검색

# 4) RAG 체인 구성 (LCEL)
prompt = ChatPromptTemplate.from_template("""다음 참고 자료만을 바탕으로 질문에 답해줘.
자료에 없는 내용이면 "자료에서 찾을 수 없습니다"라고 답해줘.

[참고 자료]
{context}

[질문]
{question}""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\\n\\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

print(rag_chain.invoke("환불 정책이 어떻게 되나요?"))
# → company_faq.txt에서 관련 내용을 찾아 그 내용에 기반해 답변


# %% [Block 2] → company_faq.txt에서 관련 내용을 찾아 그 내용에 기반해 답변 — 셀 1: 검색 결과만 먼저 확인
# 셀 1: 검색 결과만 먼저 확인
results = retriever.invoke("환불 정책")
for r in results:
    print(r.page_content[:100], "...")

# 셀 2: 프롬프트가 실제로 어떻게 채워지는지 확인
filled_prompt = prompt.invoke({"context": format_docs(results), "question": "환불 정책"})
print(filled_prompt)

# 셀 3: 전체 체인 실행
print(rag_chain.invoke("환불 정책이 어떻게 되나요?"))
