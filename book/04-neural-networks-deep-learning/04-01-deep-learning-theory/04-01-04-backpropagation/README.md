# 04-01-04 오차역전파법 — 계산 그래프로 이해하기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-04 오차역전파법 — 계산 그래프로 이해하기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_04_backpropagation.py`](04_01_04_backpropagation.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_04_backpropagation.ipynb`](04_01_04_backpropagation.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-01-04-C 덧셈 & 곱셈 계층 — "역전파의 가장 기본 부품" | 14 |
| 2 | 04-01-04-C 덧셈 & 곱셈 계층 — "역전파의 가장 기본 부품" | 12 |
| 3 | 04-01-04-C 덧셈 & 곱셈 계층 — "역전파의 가장 기본 부품" — 사과 100원 × 2개 = 200원, 소비세 1.1배 → 총 220원 | 21 |
| 4 | 04-01-04-D ReLU & Sigmoid 계층 — "활성화 함수도 역전파 가능!" | 26 |
| 5 | 04-01-04-E Affine 계층 — "신경망의 핵심 엔진" | 37 |
| 6 | 04-01-04-F Softmax-with-Loss — "역전파의 결과는 놀랍도록 간단" | 38 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-04-backpropagation
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_04_backpropagation.py
```

또는 `04_01_04_backpropagation.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
