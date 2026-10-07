# A2의 역행렬을 구하려고 하면?

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-06📐연립방정식·행렬식·역행렬.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_06_linear_systems_det_inverse.py`](01_06_linear_systems_det_inverse.py) | 코드 블록 1개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_06_linear_systems_det_inverse.ipynb`](01_06_linear_systems_det_inverse.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 01-06-E 코드로 확인하기 | 13 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-06-linear-systems-det-inverse
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_06_linear_systems_det_inverse.py
```

또는 `01_06_linear_systems_det_inverse.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 하위 절

| 번호 | 절 | 코드 블록 | 상태 |
|---|---|---|---|
| 01-06-01 | [1차식: x - 2y = 0 / 2차식: x^2 + y^2 = 5](01-06-01-solving-linear-systems/) | 3 | ✅ 실행 OK |
| 01-06-02 | [4x4 이상도 완전히 동일한 함수로 계산됩니다 — 여인수 전개가 만능인 이유](01-06-02-determinant/) | 5 | ✅ 실행 OK |
| 01-06-03 | [검산: A @ A_inv 는 반드시 단위행렬이 나와야 한다](01-06-03-inverse-matrix/) | 5 | ✅ 실행 OK |

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
