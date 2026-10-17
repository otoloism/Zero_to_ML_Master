# 1단계: 데이터 생성

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-05장 - 클러스터링과 차원축소.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_05_clustering_dim_reduction.py`](03_05_clustering_dim_reduction.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_05_clustering_dim_reduction.ipynb`](03_05_clustering_dim_reduction.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ③ 다중 실행(Multiple Runs) — scikit-learn은 기본적으로 K-Means++를 사용합니다 | 12 |
| 2 | ① NumPy로 K-평균 직접 구현 | 95 |
| 3 | ② scikit-learn으로 붓꽃 데이터 클러스터링 | 83 |
| 4 | ③ NumPy로 PCA 직접 구현 | 114 |
| 5 | ① scikit-learn PCA + 붓꽃 전체 실습 | 112 |
| 6 | ② K-평균 + PCA 결합 파이프라인 | 34 |

## 실행 방법

```bash
cd book/03-classic-ml/03-05-clustering-dim-reduction
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_05_clustering_dim_reduction.py
```

또는 `03_05_clustering_dim_reduction.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **❌ 실행 오류** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`, 2026-10-17 재검증)
- 원인: 위키독스 페이지 원문 코드를 그대로 옮겨서 생긴 문법 오류입니다 — 아래 「추출 시 수정 사항」의 ⚠️ 항목을 보세요.
- 스크립트가 멈춘 지점: `SyntaxError: invalid syntax`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/6 블록 성공

## 추출 시 수정 사항

- 위키독스 페이지([wikidocs.net/439665](https://wikidocs.net/439665))의 코드 상자를 그대로 옮겼습니다 — 블록 1개 = 코드 상자 1개, 순서·내용 동일 (2026-10-17 페이지 기준 정렬). 출력 결과·mermaid 다이어그램·글자 표(ASCII) 상자는 코드 상자로 세지 않았습니다.
- 예전에 넣었던 보충 import(`Block 0 [추가]`), 주석 처리, 오타 수정은 페이지와 1:1로 맞추기 위해 되돌렸습니다.
- ⚠️ `ff"` / `ff'` 오타(11곳)가 페이지 원문 그대로라 문법 오류가 납니다. 고치려면 위키독스 페이지를 먼저 고친 뒤 이 파일을 다시 맞춰 주세요.

## 하위 절

| 번호 | 절 | 코드 블록 | 상태 |
|---|---|---|---|
| 03-05-01 | [K-평균 알고리즘 개념과 절차](03-05-01-kmeans-concept/) | 2 | ✅ 실행 OK |
| 03-05-02 | [최적화 목표 — 왜곡 비용함수](03-05-02-distortion-cost/) | 1 | ✅ 실행 OK |
| 03-05-03 | [무작위 초기화와 지역 최적해](03-05-03-random-init-local-optima/) | 1 | ✅ 실행 OK |
| 03-05-04 | [클러스터 개수 $K$ 결정하기](03-05-04-choosing-k/) | 2 | ✅ 실행 OK |
| 03-05-05 | [K-평균 실습 — 붓꽃과 원형군집](03-05-05-kmeans-iris-circles/) | 9 | ✅ 실행 OK |
| 03-05-06 | [차원 축소와 PCA 문제 정의](03-05-06-pca-problem/) | 1 | ✅ 실행 OK |
| 03-05-07 | [PCA 알고리즘 — 전처리부터 복원까지](03-05-07-pca-algorithm/) | 1 | ✅ 실행 OK |
| 03-05-08 | [PCA 적용과 붓꽃 실습](03-05-08-pca-iris/) | 8 | 🧩 문맥 필요(조각) |

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
