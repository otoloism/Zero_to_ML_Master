# 03-03-01 특성 설계와 다항 회귀

> (Features and Polynomial Regression) — 재료를 섞어 새로운 재료를 만드는 요리사

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-03-01 특성과 다항회귀.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_03_01_features_polynomial_regression.py`](03_03_01_features_polynomial_regression.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_03_01_features_polynomial_regression.ipynb`](03_03_01_features_polynomial_regression.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 코드 — ① NumPy 직접 구현 | 61 |
| 2 | 코드 — ② scikit-learn 3줄 버전 | 22 |
| 3 | 📝 해설 | 19 |

## 실행 방법

```bash
cd book/03-classic-ml/03-03-features-classification/03-03-01-features-polynomial-regression
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_03_01_features_polynomial_regression.py
```

또는 `03_03_01_features_polynomial_regression.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
