# 병렬 실행: 요약과 번역을 동시에 수행 (순차 대비 시간 단축)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-08-04 클라우드LLM활용과 고급 체인 — 배포와 응용 패턴.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_08_04_cloud_llm_advanced_chains.py`](04_08_04_cloud_llm_advanced_chains.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_08_04_cloud_llm_advanced_chains.ipynb`](04_08_04_cloud_llm_advanced_chains.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 상세 풀이 — AWS Bedrock 예시 (LangChain 통합) | 8 |
| 2 | 📖 처음부터 끝까지 상세 풀이 | 16 |
| 3 | 조건 분기: 질문 유형에 따라 다른 체인으로 라우팅 | 21 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-08-rag-langchain/04-08-04-cloud-llm-advanced-chains
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_08_04_cloud_llm_advanced_chains.py
```

또는 `04_08_04_cloud_llm_advanced_chains.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: LLM API 패키지와 API 키 (OpenAI / Anthropic / LangChain 등) — 검증 환경에서는 설치·호출하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'langchain_aws'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/3 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
