# 03-10-01 노벨티 탐지와 아웃라이어 탐지

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-10-01 노벨티 탐지와 아웃라이어 탐지.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_10_01_novelty_outlier_detection.py`](03_10_01_novelty_outlier_detection.py) | 코드 블록 9개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_10_01_novelty_outlier_detection.ipynb`](03_10_01_novelty_outlier_detection.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ▸ ① 라이브러리 불러오기 — 수치 계산의 기본 도구 — 배열(array)을 다룹니다. | 16 |
| 2 | ▸ ② 데이터 생성 | 48 |
| 3 | ▸ ③ 눈으로 확인하기 — 산점도 | 19 |
| 4 | ▸ 직접 세어 보기 — 잡을 수 없는 이상치는 몇 개인가 | 27 |
| 5 | ▸ 직접 세어 보기 — 잡을 수 없는 이상치는 몇 개인가 — 어떤 모델이든 이 틀은 똑같습니다 | 10 |
| 6 | ▸ 정확도를 세는 방법 — 강의 자료 13p — y_pred_test 안에는 +1과 -1이 섞여 있습니다. | 13 |
| 7 | ▸ 실험 — contamination을 바꾸면 두 정확도가 어떻게 움직이나 | 24 |
| 8 | ▸ 실험 — contamination을 바꾸면 두 정확도가 어떻게 움직이나 | 16 |
| 9 | ▸ 실험 — contamination을 바꾸면 두 정확도가 어떻게 움직이나 | 13 |

## 실행 방법

```bash
cd book/03-classic-ml/03-10-anomaly-cv-practice/03-10-01-novelty-outlier-detection
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_10_01_novelty_outlier_detection.py
```

또는 `03_10_01_novelty_outlier_detection.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
