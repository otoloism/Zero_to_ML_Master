# 03-08-01 L1 노름과 L2 노름 (L1 Norm & L2 Norm)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-08-01 L1 노름과 L2 노름.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_08_01_l1_l2_norms.py`](03_08_01_l1_l2_norms.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_08_01_l1_l2_norms.ipynb`](03_08_01_l1_l2_norms.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 7단원 — NumPy로 직접 구현하기 | 21 |
| 2 | sklearn·NumPy 내장 함수와 대조 검산 — 직접 만든 함수가 맞는지, NumPy 내장 함수와 비교해 본다. | 14 |
| 3 | 규제 항으로 쓸 때의 실제 모습 — 실제 규제에서는 '가중치 벡터'의 노름을 손실에 더한다. | 16 |
| 4 | 실습 — 실습 1 정답 | 4 |
| 5 | 실습 — 실습 3 정답 — "몰빵 벡터" vs "고른 벡터" | 12 |

## 실행 방법

```bash
cd book/03-classic-ml/03-08-advanced-regularization-large-scale/03-08-01-l1-l2-norms
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_08_01_l1_l2_norms.py
```

또는 `03_08_01_l1_l2_norms.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
