# 03-04-06 신경망 비용 함수와 역전파

> (Cost Function & Back-Propagation) — 사고 현장에서 블랙박스를 역재생하며 책임을 추적하는 탐정

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-04-06 신경망 비용 함수와 역전파.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_04_06_nn_cost_backprop.py`](03_04_06_nn_cost_backprop.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_04_06_nn_cost_backprop.ipynb`](03_04_06_nn_cost_backprop.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 구현 원리 | 167 |
| 2 | 📝 해설 | 49 |

## 실행 방법

```bash
cd book/03-classic-ml/03-04-regularization-nn-basics/03-04-06-nn-cost-backprop
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_04_06_nn_cost_backprop.py
```

또는 `03_04_06_nn_cost_backprop.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
