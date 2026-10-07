# -*- coding: utf-8 -*-
"""
비공개 API (GPT)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-07장 대규모 언어 모델(LLM) 이해하기 — GPT의 시대는 무엇이 다른가/04-07-04 다양한LLM의 유형과 최신 설계 사례 — 패밀리 사진첩.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-07-04-C 3) 특화 분야에 따른 분류 — 비공개 API와 오픈소스 모델을 같은 인터페이스로 다루기 (LangChain)
# 비공개 API와 오픈소스 모델을 같은 인터페이스로 다루기 (LangChain)
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

# 비공개 API (GPT)
gpt = ChatOpenAI(model="gpt-4o-mini")

# 오픈소스 (로컬 실행) — 8.4에서 자세히 다룸
# local_llm = HuggingFacePipeline.from_model_id(model_id="meta-llama/Llama-3-8B-Instruct", task="text-generation")

print(gpt.invoke("Mixture of Experts 구조를 한 문장으로 설명해줘").content)
