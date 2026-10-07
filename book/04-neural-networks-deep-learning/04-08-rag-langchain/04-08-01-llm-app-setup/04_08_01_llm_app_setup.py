# -*- coding: utf-8 -*-
"""
OPENAI_API_KEY=sk-...

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-08장 RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기/04-08-01 LLM애플리케이션 설정 — API 모델과 로컬 오픈소스 모델.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 📖 처음부터 끝까지 상세 풀이 — .env 파일 (절대 git에 커밋하지 말 것! .gitignore에 추가)
# .env 파일 (절대 git에 커밋하지 말 것! .gitignore에 추가)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
print(llm.invoke("안녕! 너는 누구야?").content)

from langchain_anthropic import ChatAnthropic
claude = ChatAnthropic(model="claude-sonnet-4-6")
print(claude.invoke("안녕! 너는 누구야?").content)


# %% [Block 2] ANTHROPIC_API_KEY=sk-ant-... — 방법 1: Ollama (가장 간단 — 터미널에서 'ollama run llama3' 한 줄로 모델 다운로드+실행)
# 방법 1: Ollama (가장 간단 — 터미널에서 'ollama run llama3' 한 줄로 모델 다운로드+실행)
from langchain_community.llms import Ollama
local_llm = Ollama(model="llama3")
print(local_llm.invoke("안녕! 너는 누구야?"))

# 방법 2: HuggingFace Transformers (세밀한 제어 가능)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True  # 4bit 양자화로 메모리 절약
)

inputs = tokenizer("안녕! 너는 누구야?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# %% [Block 3] 방법 2: HuggingFace Transformers (세밀한 제어 가능)
from transformers import pipeline

# 텍스트 생성
generator = pipeline("text-generation", model="meta-llama/Llama-3-8B-Instruct",
                      torch_dtype="auto", device_map="auto")
result = generator("자연어 처리란", max_new_tokens=50)
print(result[0]['generated_text'])

# 한국어 요약 모델 활용
summarizer = pipeline("summarization", model="eenzeenee/t5-base-korean-summarization")
print(summarizer(long_korean_text, max_length=60))

# LangChain과 통합
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generator)
