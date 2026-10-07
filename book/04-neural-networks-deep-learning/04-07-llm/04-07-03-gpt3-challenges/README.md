# 명확한 프롬프트 (훨씬 나은 결과)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-07-03 GPT-3 사용의 도전 과제 — 천재 직원의 함정.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_07_03_gpt3_challenges.py`](04_07_03_gpt3_challenges.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_07_03_gpt3_challenges.ipynb`](04_07_03_gpt3_challenges.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-07-03-A 1) 환각(Hallucination) | 16 |
| 2 | 명확한 프롬프트 (훨씬 나은 결과) | 21 |
| 3 | 04-07-03-C 3) 컨텍스트 윈도우(Context Window) 한계 | 19 |
| 4 | 04-07-03-D 4) 비용과 지연시간(Latency) | 14 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-07-llm/04-07-03-gpt3-challenges
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_07_03_gpt3_challenges.py
```

또는 `04_07_03_gpt3_challenges.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: LLM API 패키지와 API 키 (OpenAI / Anthropic / LangChain 등) — 검증 환경에서는 설치·호출하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'openai'`
- 블록별 실행(오류가 나도 다음 블록 계속): 2/4 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
