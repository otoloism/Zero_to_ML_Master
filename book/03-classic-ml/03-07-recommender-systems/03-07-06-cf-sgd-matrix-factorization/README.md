# 03-07-06 협업 필터링 실습 — SGD 행렬 분해 (Matrix Factorization with Stochastic Gradient Descent)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-07-06 협업 필터링 실습 — SGD 행렬 분해.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_07_06_cf_sgd_matrix_factorization.py`](03_07_06_cf_sgd_matrix_factorization.py) | 코드 블록 15개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_07_06_cf_sgd_matrix_factorization.ipynb`](03_07_06_cf_sgd_matrix_factorization.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 4단원 — 행렬 준비와 초기화 | 12 |
| 2 | 4단원 — 행렬 준비와 초기화 — 잠재요인 차원 K는 3으로 가정 | 19 |
| 3 | 5단원 — RMSE : 학습 상태를 보는 계기판 | 29 |
| 4 | 6단원 — SGD 학습 루프 — 반복수, 학습률, L2 규제 | 29 |
| 5 | 7단원 — 예측 행렬 확인 | 9 |
| 6 | 7단원 — 예측 행렬 확인 — 빈칸이 어떻게 채워졌는지 보기 좋게 출력 | 8 |
| 7 | 8단원 — 재사용 가능한 함수로 묶기 | 49 |
| 8 | ① 유저-아이템 평점 행렬 만들기 — Grouplens MovieLens 데이터 | 14 |
| 9 | ② 예측 행렬 계산 — 예측 행렬 계산 | 11 |
| 10 | ③ 안 본 영화 중 예측 평점 상위 $N$편 추천 — 아직 보지 않은 영화 리스트 함수 | 30 |
| 11 | 🧪 직접 돌려보기 — 축소판 MovieLens | 27 |
| 12 | 🧪 직접 돌려보기 — 축소판 MovieLens — SGD 행렬 분해로 예측 행렬 만들기 | 10 |
| 13 | 🧪 직접 돌려보기 — 축소판 MovieLens | 20 |
| 14 | 실습 — 실습 1 정답 — K에 따른 최종 RMSE 비교 | 8 |
| 15 | 실습 — 실습 4 정답 — 순차 갱신 vs 진짜 동시 갱신 | 23 |

## 실행 방법

```bash
cd book/03-classic-ml/03-07-recommender-systems/03-07-06-cf-sgd-matrix-factorization
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_07_06_cf_sgd_matrix_factorization.py
```

또는 `03_07_06_cf_sgd_matrix_factorization.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 외부 데이터 파일 `./ml-latest-small/movies.csv` (책에 데이터 생성 코드 없음 — 직접 내려받아 같은 폴더에 두세요)
- 스크립트가 멈춘 지점: `FileNotFoundError: [Errno 2] No such file or directory: './ml-latest-small/movies.csv'`
- 블록별 실행(오류가 나도 다음 블록 계속): 12/15 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import pandas as pd` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
