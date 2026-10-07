# 검산: A @ A_inv 는 반드시 단위행렬이 나와야 한다

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-06-03🔑 역행렬(Inverse Matrix) — 행렬의 나눗셈을 완성하다.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_06_03_inverse_matrix.py`](01_06_03_inverse_matrix.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_06_03_inverse_matrix.ipynb`](01_06_03_inverse_matrix.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 🟢 3단계 — 예제① 2×2 행렬의 역행렬 구하기 | 12 |
| 2 | 🔵 4단계 — 예제② 3×3 행렬의 역행렬 구하기 | 19 |
| 3 | 🔵 5단계 — 역행렬 구하기② 행렬식·수반행렬 공식 (2×2 전용) | 17 |
| 4 | 🔴 6단계 — 역행렬이 존재하지 않는 경우 | 12 |
| 5 | 🔵 7단계 — 역행렬의 성질 | 11 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-06-linear-systems-det-inverse/01-06-03-inverse-matrix
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_06_03_inverse_matrix.py
```

또는 `01_06_03_inverse_matrix.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
