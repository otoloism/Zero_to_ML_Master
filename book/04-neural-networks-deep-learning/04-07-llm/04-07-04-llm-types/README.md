# 비공개 API (GPT)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-07-04 다양한LLM의 유형과 최신 설계 사례 — 패밀리 사진첩.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_07_04_llm_types.py`](04_07_04_llm_types.py) | 코드 블록 1개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_07_04_llm_types.ipynb`](04_07_04_llm_types.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-07-04-C 3) 특화 분야에 따른 분류 — 비공개 API와 오픈소스 모델을 같은 인터페이스로 다루기 (LangChain) | 11 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-07-llm/04-07-04-llm-types
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_07_04_llm_types.py
```

또는 `04_07_04_llm_types.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: LLM API 패키지와 API 키 (OpenAI / Anthropic / LangChain 등) — 검증 환경에서는 설치·호출하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'langchain_openai'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/1 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
