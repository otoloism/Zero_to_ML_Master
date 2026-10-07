# 01-09📉미분기초 — 순간 변화율과 경사

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-09📉미분기초 — 순간 변화율과 경사.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_09_derivative_basics.py`](01_09_derivative_basics.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_09_derivative_basics.ipynb`](01_09_derivative_basics.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 기울기 = 위로 올라간 정도(세로) ÷ 옆으로 간 정도(가로) — 기울기 = 얼마나 가파른가 = 세로 변화 ÷ 가로 변화 | 7 |
| 2 | 📊 그림 2 — 간격을 좁힐수록 "그 점의 진짜 기울기"에 다가간다 — f(x) = x²  (예: 시간 x일 때 이동거리) | 13 |
| 3 | 3단계 🟢⭐ 거듭제곱 미분 — "지수를 앞으로, 지수는 하나 줄이기" — 미분 규칙:  (x^n)' = n · x^(n-1)   ("지수를 앞으로 내리고, 지수… | 7 |
| 4 | 3단계 🟢⭐ 거듭제곱 미분 — "지수를 앞으로, 지수는 하나 줄이기" — 공식으로 구한 미분이 맞는지 '직접 재보기'로 검산 | 11 |
| 5 | 📊 그림 4 — 공이 기울기 따라 골짜기로 굴러 내려간다 — 골짜기 바닥(가장 낮은 곳) 찾기 = 기울기를 보고 아래로 걷기 | 13 |
| 6 | 📊 그림 4 — 공이 기울기 따라 골짜기로 굴러 내려간다 | 9 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-09-derivative-basics
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_09_derivative_basics.py
```

또는 `01_09_derivative_basics.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
