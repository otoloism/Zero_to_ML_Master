# 04-01-05 학습 관련 기술들 — 옵티마이저·초기화·정규화

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-05 학습 관련 기술들 — 옵티마이저·초기화·정규화.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_05_optimizers_init_regularization.py`](04_01_05_optimizers_init_regularization.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_05_optimizers_init_regularization.ipynb`](04_01_05_optimizers_init_regularization.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-01-05-A SGD의 한계 — "지그재그로 내려가는 비효율" | 11 |
| 2 | 04-01-05-B Momentum & AdaGrad — "관성 / 학습률 자동 조절" | 21 |
| 3 | 04-01-05-B Momentum & AdaGrad — "관성 / 학습률 자동 조절" | 18 |
| 4 | 04-01-05-C Adam — "실전 기본값. 일단 Adam 쓰세요" | 32 |
| 5 | 04-01-05-D 가중치 초기화 — "올바른 출발선에 서기" | 18 |
| 6 | 04-01-05-E 배치 정규화(BN) — "층마다 물 마시기" | 25 |
| 7 | 04-01-05-F 오버피팅 & 드롭아웃 — "연습만 잘하면 안 된다!" | 29 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-05-optimizers-init-regularization
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_05_optimizers_init_regularization.py
```

또는 `04_01_05_optimizers_init_regularization.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
