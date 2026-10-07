# 03-08-04 배치 · 확률적 · 미니배치 경사하강법 (Batch / Stochastic / Mini-batch Gradient Descent)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-08-04 배치 · 확률적 · 미니배치 경사하강법.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_08_04_batch_sgd_minibatch.py`](03_08_04_batch_sgd_minibatch.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_08_04_batch_sgd_minibatch.ipynb`](03_08_04_batch_sgd_minibatch.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 6단원 — 직접 구현해 비교하기 | 22 |
| 2 | 6단원 — 직접 구현해 비교하기 | 50 |
| 3 | 결과 해석 — 무엇이 승부를 갈랐나 — 배치도 충분히 오래 돌리면 결국 도달한다 — 다만 몇 배가 걸리는지 보자 | 10 |
| 4 | 실습 — 실습 1 정답 — b를 바꿔가며 1 에폭 후 성능 비교 | 12 |
| 5 | 실습 — 실습 2 정답 — 데이터가 '정렬되어' 있을 때 섞기의 효과 | 31 |
| 6 | 실습 — 실습 3 정답 — 실제 실행 시간 측정 | 18 |

## 실행 방법

```bash
cd book/03-classic-ml/03-08-advanced-regularization-large-scale/03-08-04-batch-sgd-minibatch
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_08_04_batch_sgd_minibatch.py
```

또는 `03_08_04_batch_sgd_minibatch.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
