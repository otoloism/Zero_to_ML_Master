# 03-10-07 혼동행렬과 ROC-AUC

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-10-07 혼동행렬과 ROC-AUC.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_10_07_confusion_matrix_roc_auc.py`](03_10_07_confusion_matrix_roc_auc.py) | 코드 블록 14개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_10_07_confusion_matrix_roc_auc.ipynb`](03_10_07_confusion_matrix_roc_auc.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ▸ 학습 목표 | 21 |
| 2 | ▸ 1종 오류와 2종 오류 | 41 |
| 3 | ▸ F1은 왜 산술평균이 아니라 조화평균인가 — 정밀도 1.0, 재현율 0.01인 극단적 모델을 생각해 봅시다. | 19 |
| 4 | ▸ F1은 왜 산술평균이 아니라 조화평균인가 | 14 |
| 5 | ▸ F-베타 — 균형을 직접 조절하기 | 17 |
| 6 | ▸ F-베타 — 균형을 직접 조절하기 | 23 |
| 7 | ▸ 곡선이 그려지는 과정 | 44 |
| 8 | ▸ 직접 곡선을 재현해 보기 | 38 |
| 9 | ▸ 직접 곡선을 재현해 보기 | 17 |
| 10 | ▸ AUC의 확률적 해석 — 가장 중요한 사실 | 31 |
| 11 | ▸ 이상탐지 세 모델을 AUC로 다시 비교하기 | 45 |
| 12 | ▸ ROC-AUC가 실패하는 경우 — PR 곡선 | 32 |
| 13 | ▸ 프레임워크 비교 | 30 |
| 14 | ▸ 프레임워크 비교 | 28 |

## 실행 방법

```bash
cd book/03-classic-ml/03-10-anomaly-cv-practice/03-10-07-confusion-matrix-roc-auc
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_10_07_confusion_matrix_roc_auc.py
```

또는 `03_10_07_confusion_matrix_roc_auc.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
