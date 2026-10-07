# → SQL 인젝션 취약점을 정확히 지적하는 응답

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-08-02 프롬프트 엔지니어링과GPT초기 설정 — 좋은 질문이 좋은 답을 만든다.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_08_02_prompt_engineering.py`](04_08_02_prompt_engineering.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_08_02_prompt_engineering.ipynb`](04_08_02_prompt_engineering.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-08-02-A 1) 역할 부여 (Role Prompting) | 5 |
| 2 | 04-08-02-B 2) Few-shot Prompting — 예시로 패턴 학습시키기 | 10 |
| 3 | 04-08-02-C 3) Chain-of-Thought (CoT) — 단계별 사고 유도 | 10 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-08-rag-langchain/04-08-02-prompt-engineering
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_08_02_prompt_engineering.py
```

또는 `04_08_02_prompt_engineering.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-08-01 의 코드 실행 결과 (`llm` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'llm' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/3 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
