# 03-08-02 L1 정규화(Lasso)와 L2 정규화(Ridge) (L1 & L2 Regularization)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-08-02 L1 정규화(라쏘)와 L2 정규화(릿지).md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_08_02_lasso_ridge.py`](03_08_02_lasso_ridge.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_08_02_lasso_ridge.ipynb`](03_08_02_lasso_ridge.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 식 변형 한 줄이 모든 것을 설명한다 | 17 |
| 2 | 5단원 — 숫자로 확인 : 정말 0에 닿는가 | 23 |
| 3 | 5단원 — 숫자로 확인 : 정말 0에 닿는가 — 소프트 임계(soft thresholding)를 적용한 올바른 L1 갱신 | 19 |
| 4 | 큰 가중치와 작은 가중치, 누가 먼저 죽는가 — 크기가 다른 가중치 세 개를 같은 규제로 동시에 줄여본다 | 19 |
| 5 | 실습 — 실습 1 정답 | 9 |
| 6 | 실습 — 실습 3 정답 — 10개 가중치를 L1/L2로 각각 50스텝 규제 | 18 |

## 실행 방법

```bash
cd book/03-classic-ml/03-08-advanced-regularization-large-scale/03-08-02-lasso-ridge
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_08_02_lasso_ridge.py
```

또는 `03_08_02_lasso_ridge.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
