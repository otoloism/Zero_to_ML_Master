# 04-01-03 신경망 학습 —손실 함수와 경사 하강법

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-03 신경망 학습 —손실 함수와 경사 하강법.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_03_loss_gradient_descent.py`](04_01_03_loss_gradient_descent.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_03_loss_gradient_descent.ipynb`](04_01_03_loss_gradient_descent.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-01-03-B 손실 함수 ① MSE — "(예측-정답)²의 평균" | 17 |
| 2 | 04-01-03-C 손실 함수 ② 교차 엔트로피(CEE) — "정답의 확률에 -log" | 16 |
| 3 | 04-01-03-D 미니배치 — "냄비 전체 대신 한 숟갈만 맛보기" | 15 |
| 4 | 04-01-03-E 수치 미분 — "아주 살짝 바꿔서 변화량 측정" | 12 |
| 5 | 04-01-03-F 경사 하강법 — "내리막 방향으로 한 걸음씩" | 15 |
| 6 | 04-01-03-G 학습 루프 — "5단계를 반복하라!" — 신경망 학습의 5단계 (의사 코드) | 19 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-03-loss-gradient-descent
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_03_loss_gradient_descent.py
```

또는 `04_01_03_loss_gradient_descent.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `num_epochs` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'num_epochs' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 5/6 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
