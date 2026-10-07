# 4x4 이상도 완전히 동일한 함수로 계산됩니다 — 여인수 전개가 만능인 이유

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-06-02🧮 행렬식(Determinant) — 정사각행렬 속에 숨은 넓이와 부피.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_06_02_determinant.py`](01_06_02_determinant.py) | 코드 블록 5개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_06_02_determinant.ipynb`](01_06_02_determinant.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 2단계 🟢⭐ 2×2 행렬식 — 대각선을 빼면 넓이가 나온다 | 5 |
| 2 | 4단계 🟢 3×3 계산법 ① — 대각선 법칙(사러스 법칙) | 6 |
| 3 | 5단계 🔵⭐ 3×3 계산법 ② — 여인수 전개 (모든 크기에 통하는 만능열쇠 🗝️) | 10 |
| 4 | 6단계 🟢 계산을 쉽게 만드는 두 가지 요령 | 14 |
| 5 | 7단계 🔴🚀 행렬식의 성질 — det(AB) = det(A)det(B) | 9 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-06-linear-systems-det-inverse/01-06-02-determinant
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_06_02_determinant.py
```

또는 `01_06_02_determinant.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
