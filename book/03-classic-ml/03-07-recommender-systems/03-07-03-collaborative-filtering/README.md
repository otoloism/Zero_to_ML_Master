# 03-07-03 협업 필터링 (Collaborative Filtering)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-07-03 협업 필터링.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_07_03_collaborative_filtering.py`](03_07_03_collaborative_filtering.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_07_03_collaborative_filtering.ipynb`](03_07_03_collaborative_filtering.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 6단원 — NumPy 구현 : 별점표 복원하기 | 26 |
| 2 | 6단원 — NumPy 구현 : 별점표 복원하기 | 58 |
| 3 | 6단원 — NumPy 구현 : 별점표 복원하기 | 10 |
| 4 | 6단원 — NumPy 구현 : 별점표 복원하기 | 7 |
| 5 | 6단원 — NumPy 구현 : 별점표 복원하기 — 학습된 X로 "비슷한 영화" 찾기 — 벡터 사이 거리가 가까울수록 비슷하다 | 6 |
| 6 | 프레임워크 관점 — PyTorch로 같은 일을 하면 이렇게 짧아진다 (참고용) | 20 |
| 7 | 실습 — 실습 3 정답 — R 마스크를 빼면 어떻게 되는가 | 9 |

## 실행 방법

```bash
cd book/03-classic-ml/03-07-recommender-systems/03-07-03-collaborative-filtering
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_07_03_collaborative_filtering.py
```

또는 `03_07_03_collaborative_filtering.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
