# 04-03-04 신경망 만들기 (37 51단계) —PyTorchnn Module 직접 구현

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-03-04 신경망 만들기 (37 51단계) —PyTorchnn Module 직접 구현.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_03_04_neural_net_steps_37_51.py`](04_03_04_neural_net_steps_37_51.py) | 코드 블록 14개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_03_04_neural_net_steps_37_51.ipynb`](04_03_04_neural_net_steps_37_51.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 37~38단계 — 텐서: 스칼라에서 다차원 배열로 확장 | 22 |
| 2 | 39단계 — 브로드캐스트: 크기가 다른 텐서끼리 연산 | 5 |
| 3 | 40~41단계 — 행렬 곱(MatMul): 신경망의 핵심 연산 | 8 |
| 4 | 42단계 — 선형 회귀: DeZero로 첫 번째 ML 모델! | 19 |
| 5 | 43단계 — 신경망: 2층 네트워크 구현 — 비선형 데이터: y = sin(2πx) + 노이즈 | 24 |
| 6 | 44~45단계 — Layer와 Model: PyTorch nn.Module의 정체 | 34 |
| 7 | 46단계 — Optimizer: SGD, Momentum, Adam | 27 |
| 8 | 47~48단계 — Softmax + 교차엔트로피로 다중 분류 — spiral 데이터: 나선형으로 꼬인 3개 클래스 | 17 |
| 9 | 49~50단계 — Dataset과 DataLoader: 미니배치 자동화 | 20 |
| 10 | 51단계 — MNIST 학습: DeZero로 95%+ 정확도 달성! 🎉 — 데이터 준비 | 31 |
| 11 | 04-03-04-A 지금까지는 숫자 하나(스칼라)였습니다 — Sum 함수: 여러 숫자를 하나로 더하는 연산을 자동미분 가능하게 만든 클래스 | 21 |
| 12 | 04-03-04-B 00-03의 최소제곱법을, 경사하강법 + 자동미분으로 | 26 |
| 13 | 04-03-04-C 가중치를 "특별한 변수"로 표시하기 — Parameter: Variable에 "학습 대상" 스티커를 붙인 것 (기능 동일, 타입만 다름) | 38 |
| 14 | 04-03-04-F Optimizer + 학습 루프 — PyTorch와 동일한 3줄 — SGD: 기본 경사하강법 | 18 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-03-dezero-framework/04-03-04-neural-net-steps-37-51
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_03_04_neural_net_steps_37_51.py
```

또는 `04_03_04_neural_net_steps_37_51.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-03-02 의 코드 실행 결과 (`Function` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'Function' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/14 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import numpy as np` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
