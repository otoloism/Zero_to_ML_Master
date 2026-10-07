# x1 방향으로 2만큼 떨어진 점 vs x2 방향으로 2만큼 떨어진 점

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-04 가우스 혼합 모델(GMM) — 다봉 분포 표현.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_04_gmm.py`](04_06_04_gmm.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_04_gmm.ipynb`](04_06_04_gmm.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-06-04-C 🔵 등고선으로 시각화하기 — 공분산 행렬이 만드는 타원 | 24 |
| 2 | 04-06-04-F 🔵 직접 구현하기 — GMM의 확률밀도와 책임 계산 | 34 |
| 3 | 04-06-04-G 🟢 sklearn으로 실습하기 — 2D 데이터(3개 봉우리) 학습 | 20 |
| 4 | 04-06-04-K 📝 연습문제 | 6 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-04-gmm
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_04_gmm.py
```

또는 `04_06_04_gmm.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
