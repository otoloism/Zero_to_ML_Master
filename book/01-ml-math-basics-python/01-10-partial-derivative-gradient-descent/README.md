# 01-10🧭편미분· 그래디언트 ·경사하강법

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `01-10🧭편미분· 그래디언트 ·경사하강법.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`01_10_partial_derivative_gradient_descent.py`](01_10_partial_derivative_gradient_descent.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`01_10_partial_derivative_gradient_descent.ipynb`](01_10_partial_derivative_gradient_descent.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 01-10-C 3단계 — 🧩 벽 ③: 미분과 벡터가 한꺼번에 나온다 | 31 |
| 2 | 01-10-E 5단계 — 🙈 벽 ⑤: 지도를 볼 수 없다는 사실 | 37 |
| 3 | 01-10-E 5단계 — 🙈 벽 ⑤: 지도를 볼 수 없다는 사실 | 21 |
| 4 | 01-10-F 6단계 — 👣 벽 ⑥: 보폭(학습률)의 딜레마 | 6 |
| 5 | 01-10-G 7단계 — 🕳️ 벽 ⑦: 발밑이 평평하다고 바닥은 아니다 | 13 |
| 6 | 01-10-H 8단계 — 🚀 신경망 학습 = 수백만 차원 경사하강법 | 3 |
| 7 | 📝 연습문제 | 4 |
| 8 | 📝 연습문제 | 9 |

## 실행 방법

```bash
cd book/01-ml-math-basics-python/01-10-partial-derivative-gradient-descent
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 01_10_partial_derivative_gradient_descent.py
```

또는 `01_10_partial_derivative_gradient_descent.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: Block 6 은 PyTorch 학습 루프 3줄 요약(`optimizer.zero_grad()` 등)으로, 앞 블록에서 만든 직접 구현 옵티마이저에는 해당 메서드가 없어 오류가 납니다 (설명용 조각).
- 블록별 실행(오류가 나도 다음 블록 계속): 7/8 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
