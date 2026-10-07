# 03-05-08 PCA 적용과 붓꽃 실습

> (Applying PCA - Speedup, Misuse, and the Iris Pipeline) - 좋은 도구일수록 쓸 자리를 가려야 한다

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-05-08 PCA 적용과 붓꽃 실습.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_05_08_pca_iris.py`](03_05_08_pca_iris.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_05_08_pca_iris.ipynb`](03_05_08_pca_iris.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 파이프라인으로 안전하게 | 18 |
| 2 | 코드 — 1. 데이터 로드 및 데이터 파악 — 실습 데이터셋 라이브러리. | 38 |
| 3 | 코드 — 2. 기술통계량(데이터 분포) 확인 — 결측치 여부 파악 | 16 |
| 4 | 코드 — sklearn으로 PC score 구하기 — PCA 함수를 활용하여 PC를 얻어낸다. | 12 |
| 5 | 코드 — 직접 계산으로 검산하기 — PC score 구하기 | 15 |
| 6 | 시각화 ③ — 주성분 방향 그려보기 — PC score scatter | 13 |
| 7 | 코드 — 4개 특성 전부로 PCA | 10 |
| 8 | 코드 — 로지스틱 회귀와 혼동 행렬 — ✅ 최신 환경 수정본 | 25 |

## 실행 방법

```bash
cd book/03-classic-ml/03-05-clustering-dim-reduction/03-05-08-pca-iris
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_05_08_pca_iris.py
```

또는 `03_05_08_pca_iris.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `X_train` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'X_train' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 6/8 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
