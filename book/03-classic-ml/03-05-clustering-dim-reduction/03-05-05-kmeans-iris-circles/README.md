# 03-05-05 K-평균 실습 — 붓꽃과 원형군집

> (K-Means Practice with Iris & Blobs) - 이론이 처음으로 화면 위에서 움직이는 순간

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-05-05 K-평균 실습 — 붓꽃과 원형군집.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_05_05_kmeans_iris_circles.py`](03_05_05_kmeans_iris_circles.py) | 코드 블록 9개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_05_05_kmeans_iris_circles.ipynb`](03_05_05_kmeans_iris_circles.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 코드 — library | 35 |
| 2 | 코드 (강의 원본 → 최신 환경 수정본) — ✅ 최신 환경 수정본 | 15 |
| 3 | 코드 — labels_ : 각 데이터가 몇 번 군집인지 (150개짜리 정수 배열) | 30 |
| 4 | 🚨 가장 큰 함정 — "정확도"를 계산하려는 순간 — ✅ 올바른 방법 : 교차표(crosstab)로 대응 관계를 먼저 확인한다 | 8 |
| 5 | 코드 — 데이터 생성 | 21 |
| 6 | 코드 — 학습과 라벨 확인 | 13 |
| 7 | 코드 — 군집별 색·모양과 중심점 별표 — X[y_km == 0, 0] 의 뜻을 천천히 뜯어봅시다. | 34 |
| 8 | 5단원 — K=4로 늘리면 생기는 일 | 12 |
| 9 | 6단원 — 엘보우 곡선으로 마무리 | 16 |

## 실행 방법

```bash
cd book/03-classic-ml/03-05-clustering-dim-reduction/03-05-05-kmeans-iris-circles
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_05_05_kmeans_iris_circles.py
```

또는 `03_05_05_kmeans_iris_circles.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
