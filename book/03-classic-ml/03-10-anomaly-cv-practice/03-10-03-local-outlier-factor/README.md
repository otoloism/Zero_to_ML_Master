# 그림을 그리려면 아래 코드를 실행하세요

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-10-03 Local Outlier Factor — 우리 동네와 옆 동네의 밀도 비교.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_10_03_local_outlier_factor.py`](03_10_03_local_outlier_factor.py) | 코드 블록 9개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_10_03_local_outlier_factor.ipynb`](03_10_03_local_outlier_factor.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ▸ 숫자로 증명하기 | 49 |
| 2 | ▸ 전체 흐름 한눈에 보기 | 57 |
| 3 | ▸ 결과 해석 | 22 |
| 4 | ▸ 결과 해석 | 4 |
| 5 | ▸ 왜 좋아졌는가 — 그 빈 공간을 다시 물어보기 | 17 |
| 6 | ▸ 결과 시각화 — 강의 자료 16p | 22 |
| 7 | ▸ ① novelty — 하나의 클래스, 두 개의 인격 | 37 |
| 8 | ▸ ② n_neighbors — 몇 명까지 이웃으로 볼 것인가 | 24 |
| 9 | ▸ 프레임워크 비교 | 26 |

## 실행 방법

```bash
cd book/03-classic-ml/03-10-anomaly-cv-practice/03-10-03-local-outlier-factor
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_10_03_local_outlier_factor.py
```

또는 `03_10_03_local_outlier_factor.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 03-10-01 의 코드 실행 결과 (`outliers` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'outliers' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 3/9 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
