# OPENAI_API_KEY=sk-...

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-08-01 LLM애플리케이션 설정 — API 모델과 로컬 오픈소스 모델.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_08_01_llm_app_setup.py`](04_08_01_llm_app_setup.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_08_01_llm_app_setup.ipynb`](04_08_01_llm_app_setup.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |
| [`commands.sh`](commands.sh) | 책에 나온 셸 명령(설치 등) |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 상세 풀이 — .env 파일 (절대 git에 커밋하지 말 것! .gitignore에 추가) | 14 |
| 2 | ANTHROPIC_API_KEY=sk-ant-... — 방법 1: Ollama (가장 간단 — 터미널에서 'ollama run llama3' 한 줄로 모델 다운… | 18 |
| 3 | 방법 2: HuggingFace Transformers (세밀한 제어 가능) | 15 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-08-rag-langchain/04-08-01-llm-app-setup
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_08_01_llm_app_setup.py
```

또는 `04_08_01_llm_app_setup.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: LLM API 패키지와 API 키 (OpenAI / Anthropic / LangChain 등) — 검증 환경에서는 설치·호출하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'dotenv'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/3 블록 성공

## 셸 명령

```bash
pip install langchain langchain-openai langchain-community python-dotenv transformers accelerate
```

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
