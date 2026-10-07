# 03-07-02 콘텐츠 기반 추천 시스템 (Content Based Recommendation)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-07-02 콘텐츠 기반 추천 시스템.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_07_02_content_based_recsys.py`](03_07_02_content_based_recsys.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_07_02_content_based_recsys.ipynb`](03_07_02_content_based_recsys.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 직관 — 왜 곱해서 더하나 | 10 |
| 2 | 5단원 — NumPy 구현 : Alice의 취향을 학습시키기 | 19 |
| 3 | 5단원 — NumPy 구현 : Alice의 취향을 학습시키기 | 58 |
| 4 | 코드 설명 — 결과 읽는 법 — 학습한 취향으로 안 본 영화(2번) 예측 | 7 |
| 5 | 코드 설명 — 결과 읽는 법 — 정규화 세기(lambda)에 따라 취향이 어떻게 변하는지 비교 | 7 |
| 6 | 6단원 — 사용자 4명 전체로 확장 — 사용자 4명의 별점표 (행=영화, 열=사용자). 안 본 칸은 nan | 29 |
| 7 | 프레임워크 관점 | 9 |
| 8 | 실습 — 실습 4 미리 확인 — 아무것도 평가하지 않은 사용자 Eve | 6 |

## 실행 방법

```bash
cd book/03-classic-ml/03-07-recommender-systems/03-07-02-content-based-recsys
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_07_02_content_based_recsys.py
```

또는 `03_07_02_content_based_recsys.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
