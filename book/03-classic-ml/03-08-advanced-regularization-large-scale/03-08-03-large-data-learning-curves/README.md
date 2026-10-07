# 03-08-03 대규모 데이터와 학습 곡선 (Learning with Large Datasets & Learning Curve)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-08-03 대규모 데이터와 학습 곡선.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_08_03_large_data_learning_curves.py`](03_08_03_large_data_learning_curves.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_08_03_large_data_learning_curves.ipynb`](03_08_03_large_data_learning_curves.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 6단원 — 코드로 직접 그려 확인하기 | 42 |
| 2 | 6단원 — 코드로 직접 그려 확인하기 | 8 |
| 3 | 실습 — 실습 1 정답 — 3차 모델(정답과 같은 복잡도) | 9 |
| 4 | 실습 — 실습 3 정답 — 9차 모델에 Ridge 규제를 걸면 과대적합이 잡힐까? | 38 |

## 실행 방법

```bash
cd book/03-classic-ml/03-08-advanced-regularization-large-scale/03-08-03-large-data-learning-curves
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_08_03_large_data_learning_curves.py
```

또는 `03_08_03_large_data_learning_curves.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
