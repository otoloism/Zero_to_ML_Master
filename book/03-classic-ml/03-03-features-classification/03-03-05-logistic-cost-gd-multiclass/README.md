# 03-03-05 비용 함수·경사하강법·다중 클래스 분류

> (Cost Function, Gradient Descent & Multi-Class Classification) — 로그가 만드는 그릇, 그리고 셋 이상 나누기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-03-05 비용 함수·경사하강법·다중 클래스 분류.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_03_05_logistic_cost_gd_multiclass.py`](03_03_05_logistic_cost_gd_multiclass.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_03_05_logistic_cost_gd_multiclass.ipynb`](03_03_05_logistic_cost_gd_multiclass.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 구현 원리 | 105 |
| 2 | 코드 — One-vs-All 직접 구현 + sklearn 대조 | 63 |
| 3 | 📝 해설 | 26 |

## 실행 방법

```bash
cd book/03-classic-ml/03-03-features-classification/03-03-05-logistic-cost-gd-multiclass
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_03_05_logistic_cost_gd_multiclass.py
```

또는 `03_03_05_logistic_cost_gd_multiclass.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
