# 우리는 GPU도, 학습 데이터도, 파라미터 튜닝도 필요 없습니다.

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-07-02 LLM을 개발·활용하는 이유와 개발 도전 과제 — 거대한 공장을 짓는 이유.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_07_02_why_llm_challenges.py`](04_07_02_why_llm_challenges.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_07_02_why_llm_challenges.ipynb`](04_07_02_why_llm_challenges.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 🟢 1단계 — 대규모 언어 모델을 "개발"하고 "활용"하는 이유 — "개발"이 아닌 "활용" — 이미 지어진 공장(모델)에 주문만 넣는 코드 | 16 |
| 2 | 🔴 2단계 — 대규모 언어 모델 개발 도전 과제 — Scaling Law 개념 시각화 (실제 학습이 아닌 개념 그래프) | 18 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-07-llm/04-07-02-why-llm-challenges
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_07_02_why_llm_challenges.py
```

또는 `04_07_02_why_llm_challenges.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `call_api` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'call_api' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
