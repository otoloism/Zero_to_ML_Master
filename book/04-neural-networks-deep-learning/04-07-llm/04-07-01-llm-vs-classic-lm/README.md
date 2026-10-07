# API 키는 환경변수로 등록 (자세한 설정은 04-08장에서 다룹니다)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-07-01 LLM과 기존 언어 모델의 차이 — _전문가 한 명_과 _백과사전 박사_.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_07_01_llm_vs_classic_lm.py`](04_07_01_llm_vs_classic_lm.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_07_01_llm_vs_classic_lm.ipynb`](04_07_01_llm_vs_classic_lm.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |
| [`commands.sh`](commands.sh) | 책에 나온 셸 명령(설치 등) |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | API 키는 환경변수로 등록 (자세한 설정은 04-08장에서 다룹니다) | 18 |
| 2 | 4) 새로운 문장 예측 | 13 |
| 3 | "중립" — 03-99 부록에서 수백 개 데이터로 모델을 학습시켜야 했던 작업을, 학습 없이 즉시 수행 | 13 |
| 4 | "중립" — 03-99 부록에서 수백 개 데이터로 모델을 학습시켜야 했던 작업을, 학습 없이 즉시 수행 | 13 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-07-llm/04-07-01-llm-vs-classic-lm
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_07_01_llm_vs_classic_lm.py
```

또는 `04_07_01_llm_vs_classic_lm.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: LLM API 패키지와 API 키 (OpenAI / Anthropic / LangChain 등) — 검증 환경에서는 설치·호출하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'openai'`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/4 블록 성공

## 셸 명령

```bash
pip install openai anthropic scikit-learn matplotlib
# API 키는 환경변수로 등록 (자세한 설정은 04-08장에서 다룹니다)
export OPENAI_API_KEY="여러분의_키"
export ANTHROPIC_API_KEY="여러분의_키"
```

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
