# 03-08-06 Ridge · Lasso · ElasticNet 실습 (Regularization Practice)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-08-06 Ridge · Lasso · ElasticNet 실습.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_08_06_ridge_lasso_elasticnet_practice.py`](03_08_06_ridge_lasso_elasticnet_practice.py) | 코드 블록 18개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_08_06_ridge_lasso_elasticnet_practice.ipynb`](03_08_06_ridge_lasso_elasticnet_practice.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 데이터셋에 관한 안내 | 31 |
| 2 | 2단원 — 여러 모델을 한 표에 모으는 도구 만들기 — 모델들의 성적을 모아두는 상자 | 33 |
| 3 | 3단원 — Ridge (L2 Regularization) — 값이 커질 수록 큰 규제입니다. | 11 |
| 4 | 4단원 — coef_ 읽기 : 어떤 특성이 중요한가 | 24 |
| 5 | 5단원 — Ridge : alpha에 따른 계수의 변화 — 규제가 아주 강한 모델과 아주 약한 모델을 각각 만든다 | 11 |
| 6 | 5단원 — Ridge : alpha에 따른 계수의 변화 | 2 |
| 7 | 두 결과를 나란히 놓고 비교 — 계수의 '크기'가 얼마나 줄었는지 숫자로 확인 | 14 |
| 8 | 6단원 — Lasso (L1 Regularization) — 값이 커질 수록 큰 규제입니다. | 10 |
| 9 | 7단원 — 하이라이트 : 계수가 0이 되는 순간 | 18 |
| 10 | 살아남은 특성은 무엇인가 — alpha=1 에서 살아남은 3개가 무엇인지 확인한다 | 10 |
| 11 | 살아남은 특성은 무엇인가 — 마지막으로 Ridge와 Lasso를 같은 alpha에서 정면 비교한다 | 17 |
| 12 | 8단원 — ElasticNet : 두 규제를 섞기 | 11 |
| 13 | 8단원 — ElasticNet : 두 규제를 섞기 — l1_ratio가 0에서 1로 갈 때 무슨 일이 일어나는지 촘촘히 본다 | 19 |
| 14 | 8단원 — ElasticNet : 두 규제를 섞기 — ElasticNet의 계수도 확인해 본다 (강의 자료는 alpha=5로 두 개를 비교했다) | 14 |
| 15 | 9단원 — 총정리 : 무엇을 어떻게 고를까 — 감으로 고르지 말고 교차검증으로 고르는 방법 | 29 |
| 16 | 실습 — 실습 1 정답 — 첫 계수가 죽는 alpha 찾기 | 11 |
| 17 | 실습 — 실습 2 정답 — 잡음 특성 20개를 섞고 Lasso가 걸러내는지 본다 | 37 |
| 18 | 실습 — 실습 3 정답 — alpha와 l1_ratio를 동시에 탐색 | 17 |

## 실행 방법

```bash
cd book/03-classic-ml/03-08-advanced-regularization-large-scale/03-08-06-ridge-lasso-elasticnet-practice
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_08_06_ridge_lasso_elasticnet_practice.py
```

또는 `03_08_06_ridge_lasso_elasticnet_practice.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
