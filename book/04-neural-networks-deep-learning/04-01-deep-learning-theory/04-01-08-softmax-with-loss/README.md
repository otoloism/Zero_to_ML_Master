# 04-01-08 Softmax-with-Loss 계층의 계산 그래프 유도

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-08 Softmax-with-Loss 계층의 계산 그래프 유도.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_08_softmax_with_loss.py`](04_01_08_softmax_with_loss.py) | 코드 블록 9개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_08_softmax_with_loss.ipynb`](04_01_08_softmax_with_loss.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 친절하게 풀어보기 — 1) for문: 같은 작업을 여러 번 반복하기 | 33 |
| 2 | 04-01-08-A 깊이에는 두 얼굴이 있습니다 — 전화기 놀이를 떠올려 보세요 — 시그모이드 함수의 미분값은 아무리 커도 0.25입니다. | 13 |
| 3 | 04-01-08-C 덧셈 노드는 신호를 그대로 나눠주는 "만능 배달원" — 덧셈 y = a + b 의 역전파: | 19 |
| 4 | 04-01-08-D 코드로 지름길의 효과를 직접 확인해 봅시다 | 39 |
| 5 | 04-01-08-E VGGNet — "작은 레고 블록만으로 단순하게 쌓기" — 방법 A: 5×5 필터 한 번 쓰기 | 22 |
| 6 | 04-01-08-F GoogLeNet — "세로 말고 가로로도 넓혀보자 (Inception 모듈)" | 35 |
| 7 | 04-01-08-EF VGGNet vs GoogLeNet vs ResNet — 한눈에 비교 — 유명 구조들의 파라미터 수와 층 수를 비교합니다 | 20 |
| 8 | 04-01-08-G 왜 GPU가 딥러닝에 잘 맞을까요? — 일개미 부대 비유 | 31 |
| 9 | 📖 유도 — 결과부터 먼저, 그다음 한 줄씩 | 28 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-08-softmax-with-loss
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_08_softmax_with_loss.py
```

또는 `04_01_08_softmax_with_loss.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
