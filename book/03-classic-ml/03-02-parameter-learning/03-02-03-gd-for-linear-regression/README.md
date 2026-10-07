# 03-02-03 선형회귀를 위한 경사하강법 — 수식 유도

> 오차라는 언덕의 기울기를 미분으로 재는 법

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-02-03 선형회귀를 위한 경사하강법 - 수식 유도.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_02_03_gd_for_linear_regression.py`](03_02_03_gd_for_linear_regression.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_02_03_gd_for_linear_regression.ipynb`](03_02_03_gd_for_linear_regression.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | gradients.py | 43 |
| 2 | gradient_descent_linear.py | 42 |
| 3 | multivariate_gd.py | 48 |
| 4 | feature_scaling.py | 39 |
| 5 | 다. 구현 문제 — 벡터화 방식 | 13 |

## 실행 방법

```bash
cd book/03-classic-ml/03-02-parameter-learning/03-02-03-gd-for-linear-regression
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_02_03_gd_for_linear_regression.py
```

또는 `03_02_03_gd_for_linear_regression.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `x_new` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'x_new' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 4/5 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
