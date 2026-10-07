# 01-05행렬내적·외적·유사도 행렬

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-05행렬내적·외적·유사도 행렬.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_05_dot_outer_similarity_matrix.py`](01_05_dot_outer_similarity_matrix.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_05_dot_outer_similarity_matrix.ipynb`](01_05_dot_outer_similarity_matrix.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 1단계 🟢⭐ 내적의 두 얼굴 — 계산식과 각도식은 왜 같은가 \[기계적 계산 격파\] | 14 |
| 2 | 2단계 🟢⭐ 크기의 함정 — 내적이 거짓말을 할 때 \[크기의 혼란 격파\] | 15 |
| 3 | 3단계 🔵⭐ 벡터 외적(cross) — 숫자가 아니라 방향이 나온다 \[차원 혼란 격파\] | 14 |
| 4 | 4단계 🔵⭐ 외적(outer) — 같은 이름, 완전히 다른 연산 \[정의 혼재 격파\] | 12 |
| 5 | 5단계 🔵⭐ 유사도 행렬 $XX^{T}$ — 모든 쌍을 한 번에 \[관계 압축 격파\] | 13 |
| 6 | 6단계 🔵 코사인 유사도 행렬 — 추천 시스템을 굴려 보다 | 15 |
| 7 | 7단계 🔵🚀 어텐션 — AI는 왜 이 거대한 표를 만드는가 \[응용 복잡성 격파\] | 24 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-05-dot-outer-similarity-matrix
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_05_dot_outer_similarity_matrix.py
```

또는 `01_05_dot_outer_similarity_matrix.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 하위 절

| 번호 | 절 | 코드 블록 | 상태 |
|---|---|---|---|
| 01-05-01 | [0차원 텐서 (스칼라) — 축이 없음](01-05-01-tensors/) | 3 | ✅ 실행 OK |

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
