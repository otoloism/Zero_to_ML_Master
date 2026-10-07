# 03-09-01 부스팅 알고리즘 부스팅의 개념과 배깅과의 차이

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-09-01 부스팅 알고리즘 부스팅의 개념과 배깅과의 차이.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_09_01_boosting_vs_bagging.py`](03_09_01_boosting_vs_bagging.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_09_01_boosting_vs_bagging.ipynb`](03_09_01_boosting_vs_bagging.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 구현 관점 — "약한 학습기 하나"와 "강한 학습기 하나"를 실제로 비교해 봅니다. | 29 |
| 2 | 직관적 의미 — 왜 부스팅은 편향을 줄이나 — 트리를 하나씩 더할 때 훈련/시험 오차가 어떻게 변하는지 직접 봅니다. | 31 |
| 3 | 수학적 의미 — 라이브러리 없이 만드는 미니 그레이디언트 부스팅 (트리 3그루) | 41 |
| 4 | 검증 — 사이킷런 결과와 같은가? — 우리가 손으로 만든 3그루 부스팅과 | 13 |
| 5 | 실습 — [실습 2 정답 예시] 30그루까지 늘려 남은 오차를 추적합니다. | 15 |

## 실행 방법

```bash
cd book/03-classic-ml/03-09-boosting/03-09-01-boosting-vs-bagging
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_09_01_boosting_vs_bagging.py
```

또는 `03_09_01_boosting_vs_bagging.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 추출 시 수정 사항

- 이중 줄바꿈(추출 아티팩트) 정리

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
