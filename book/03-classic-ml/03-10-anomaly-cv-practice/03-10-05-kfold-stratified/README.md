# 교차 검증:1

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-10-05 K-Fold와 Stratified K-Fold.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_10_05_kfold_stratified.py`](03_10_05_kfold_stratified.py) | 코드 블록 14개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_10_05_kfold_stratified.ipynb`](03_10_05_kfold_stratified.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ▸ 학습 목표 | 27 |
| 2 | ▸ 학습 목표 | 25 |
| 3 | ▸ 직관 — 왜 표준편차까지 봐야 하나 | 18 |
| 4 | ▸ 직관 — 왜 표준편차까지 봐야 하나 | 34 |
| 5 | ▸ 표준편차까지 함께 보기 | 14 |
| 6 | ▸ 표준편차까지 함께 보기 | 25 |
| 7 | ▸ 범인 찾기 — 레이블 분포를 들여다보기 | 29 |
| 8 | ▸ 왜 5-폴드에서는 안 터졌을까 | 15 |
| 9 | ▸ 시각화 ② — 계층적 분할이 하는 일 | 21 |
| 10 | ▸ 성능 재측정 — 강의 자료 31p | 28 |
| 11 | ▸ KFold vs StratifiedKFold 최종 비교 | 33 |
| 12 | ▸ KFold vs StratifiedKFold 최종 비교 | 22 |
| 13 | ▸ 프레임워크 비교 | 23 |
| 14 | ▸ 프레임워크 비교 | 33 |

## 실행 방법

```bash
cd book/03-classic-ml/03-10-anomaly-cv-practice/03-10-05-kfold-stratified
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_10_05_kfold_stratified.py
```

또는 `03_10_05_kfold_stratified.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
