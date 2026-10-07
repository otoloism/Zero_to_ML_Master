# 03-09-02 부스팅 알고리즘 AdaBoost — 적응형 부스팅

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-09-02 부스팅 알고리즘 AdaBoost — 적응형 부스팅.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_09_02_adaboost.py`](03_09_02_adaboost.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_09_02_adaboost.ipynb`](03_09_02_adaboost.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 3단계 — 손으로 굴리는 3라운드 — 가중치는 어디로 움직이나 — AdaBoost를 라이브러리 없이 손으로 3라운드 굴려 봅니다. | 91 |
| 2 | 세 그루터기를 합치면? — 세 그루터기를 α 로 가중 합해 최종 예측을 만듭니다. | 21 |
| 3 | 4단계 — 사이킷런 AdaBoostClassifier 실습 — 사이킷런 AdaBoostClassifier 기본 사용법 | 32 |
| 4 | 5단계 — 학습률과 트리 수의 시소 관계 — 학습률과 학습기 개수를 바꿔 가며 시험 정확도를 비교합니다. | 20 |
| 5 | 실습 — [실습 3 정답 예시] 라운드를 늘리며 훈련 정확도를 추적합니다. | 26 |

## 실행 방법

```bash
cd book/03-classic-ml/03-09-boosting/03-09-02-adaboost
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_09_02_adaboost.py
```

또는 `03_09_02_adaboost.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 추출 시 수정 사항

- 이중 줄바꿈(추출 아티팩트) 정리

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
