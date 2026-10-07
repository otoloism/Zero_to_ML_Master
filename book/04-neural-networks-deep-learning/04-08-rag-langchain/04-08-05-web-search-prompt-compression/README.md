# 내부적으로: 1) "검색이 필요하다" 판단 → 2) web_search 호출 → 3) 결과를 바탕으로 답변 생성

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-08-05 웹 검색 자동화와프롬프트 압축— 최신 정보와 비용 절감.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_08_05_web_search_prompt_compression.py`](04_08_05_web_search_prompt_compression.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_08_05_web_search_prompt_compression.ipynb`](04_08_05_web_search_prompt_compression.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 상세 풀이 | 19 |
| 2 | 내부적으로: 1) "검색이 필요하다" 판단 → 2) web_search 호출 → 3) 결과를 바탕으로 답변 생성 — 1) 대화 요약을 통한 압축 | 25 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-08-rag-langchain/04-08-05-web-search-prompt-compression
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_08_05_web_search_prompt_compression.py
```

또는 `04_08_05_web_search_prompt_compression.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: LLM API 패키지와 API 키 (OpenAI / Anthropic / LangChain 등) — 검증 환경에서는 설치·호출하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'langchain'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
