# 04-03-05 DeZero의 도전 (52 60단계) — GPU·CNN·RNN까지

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-03-05 DeZero의 도전 (52 60단계) — GPU·CNN·RNN까지.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_03_05_gpu_cnn_rnn_steps_52_60.py`](04_03_05_gpu_cnn_rnn_steps_52_60.py) | 코드 블록 10개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_03_05_gpu_cnn_rnn_steps_52_60.ipynb`](04_03_05_gpu_cnn_rnn_steps_52_60.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 52 GPU 지원 — "일반 도로에서 고속도로로 차선 변경" | 31 |
| 2 | 53 모델 저장/읽기 — "게임 세이브/로드" | 30 |
| 3 | 54 Dropout — "랜덤 결석 훈련법" | 25 |
| 4 | 55 CNN 메커니즘 — "돋보기로 사진 훑기" — 55단계: 합성곱의 직관적 구현 (이해용) | 32 |
| 5 | 56 im2col 함수 — "퍼즐 조각을 일렬로 세우기" | 19 |
| 6 | 57 conv2d / pooling — "CNN 연산을 Function으로 포장" | 41 |
| 7 | 58 VGG16 구현 — "16층 고층 빌딩 세우기" | 37 |
| 8 | 59 RNN 시계열 처리 — "순서를 기억하는 신경망" | 24 |
| 9 | 60 LSTM과 데이터 로더 — "장기 기억 노트 + 자동 급식" | 40 |
| 10 | 60 LSTM과 데이터 로더 — "장기 기억 노트 + 자동 급식" | 44 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-03-dezero-framework/04-03-05-gpu-cnn-rnn-steps-52-60
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_03_05_gpu_cnn_rnn_steps_52_60.py
```

또는 `04_03_05_gpu_cnn_rnn_steps_52_60.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-03-04 의 코드 실행 결과 (`model` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'model' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 4/10 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import numpy as np` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
