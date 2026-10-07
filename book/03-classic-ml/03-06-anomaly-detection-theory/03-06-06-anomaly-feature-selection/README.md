# 03-06-06 이상탐지에서의 Feature 선택

> (Choosing Features for Anomaly Detection) - 알고리즘보다 재료가 성능을 결정한다

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-06-06 이상탐지에서의 Feature 선택.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_06_06_anomaly_feature_selection.py`](03_06_06_anomaly_feature_selection.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_06_06_anomaly_feature_selection.ipynb`](03_06_06_anomaly_feature_selection.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 코드 | 40 |
| 2 | 실험 — 비율 특성 하나가 만드는 차이 | 58 |

## 실행 방법

```bash
cd book/03-classic-ml/03-06-anomaly-detection-theory/03-06-06-anomaly-feature-selection
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_06_06_anomaly_feature_selection.py
```

또는 `03_06_06_anomaly_feature_selection.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
