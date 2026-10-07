# 04-01-02 신경망 —활성화 함수와 순전파

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-02 신경망 —활성화 함수와 순전파.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_02_activation_forward.py`](04_01_02_activation_forward.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_02_activation_forward.ipynb`](04_01_02_activation_forward.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-01-02-B AND, NAND, OR 게이트 — "퍼셉트론의 첫 번째 임무" | 34 |
| 2 | 04-01-02-C XOR — "직선 하나로는 풀 수 없는 문제" | 10 |
| 3 | 04-01-02-D 활성화 함수 ① 시그모이드 — "계단을 S자 곡선으로 교체" | 21 |
| 4 | 04-01-02-E 활성화 함수 ② ReLU — "음수는 차단, 양수는 그대로" | 10 |
| 5 | 04-01-02-F 3층 신경망 순전파 — "공장 조립 라인 가동!" | 48 |
| 6 | 04-01-02-G 소프트맥스 — "점수를 확률로 바꾸기" | 16 |
| 7 | 04-01-02-H MNIST 손글씨 인식 — "진짜 이미지를 분류하는 신경망" | 38 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-02-activation-forward
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_02_activation_forward.py
```

또는 `04_01_02_activation_forward.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
