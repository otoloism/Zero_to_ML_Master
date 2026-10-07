# 03-07-04 저차원 행렬분해와 평균 정규화 (Low Rank Matrix Factorization & Mean Normalization)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-07-04 저차원 행렬분해와 평균 정규화.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_07_04_low_rank_mean_normalization.py`](03_07_04_low_rank_mean_normalization.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_07_04_low_rank_mean_normalization.ipynb`](03_07_04_low_rank_mean_normalization.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 2단원 — 왜 '저차원(Low Rank)'인가 | 18 |
| 2 | 4단원 — 신규 사용자 Eve 문제 — Eve 문제를 직접 눈으로 확인 | 34 |
| 3 | 강의 슬라이드 43쪽 계산 그대로 확인하기 — 각 영화의 평균 별점 mu (nan은 빼고 계산) | 18 |
| 4 | 강의 슬라이드 43쪽 계산 그대로 확인하기 — 평균을 뺀 표로 학습 | 18 |
| 5 | 6단원 — 종합 구현 : 추천 파이프라인 한 벌 | 37 |
| 6 | 6단원 — 종합 구현 : 추천 파이프라인 한 벌 — "이 영화와 비슷한 작품" 기능 | 13 |
| 7 | 실습 — 실습 3 정답 — 아무도 평가하지 않은 신규 영화 | 10 |

## 실행 방법

```bash
cd book/03-classic-ml/03-07-recommender-systems/03-07-04-low-rank-mean-normalization
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_07_04_low_rank_mean_normalization.py
```

또는 `03_07_04_low_rank_mean_normalization.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
