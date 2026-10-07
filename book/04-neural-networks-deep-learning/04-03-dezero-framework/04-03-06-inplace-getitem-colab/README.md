# 04-03-06 인플레이스 연산 · get_item 함수 · 구글 콜랩

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-03-06 인플레이스 연산 · get_item 함수 · 구글 콜랩.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_03_06_inplace_getitem_colab.py`](04_03_06_inplace_getitem_colab.py) | 코드 블록 11개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_03_06_inplace_getitem_colab.ipynb`](04_03_06_inplace_getitem_colab.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-03-06-A 52단계 — GPU 지원: "같은 코드, 다른 백엔드" — 52단계: NumPy ↔ CuPy 자동 전환 | 59 |
| 2 | 04-03-06-B 53단계 — 모델 저장/읽기: "시험 공부 결과를 파일에 적기" | 31 |
| 3 | 04-03-06-C 54단계 — Dropout: "학습 때만 일부러 일부 뉴런을 끄기" | 35 |
| 4 | 04-03-06-D 55~57단계 — CNN: im2col과 Conv2d | 54 |
| 5 | 04-03-06-E 55단계 — Pooling: "각 구역에서 가장 강한 신호만 남기기" | 41 |
| 6 | 04-03-06-F 58단계 — VGG16: "3×3 레고 블록을 16층으로 쌓기" | 40 |
| 7 | 04-03-06-G 59단계 — RNN: "매일 일기장을 이어 쓰는 신경망" | 69 |
| 8 | 04-03-06-H 60단계 — LSTM: "3개의 문이 달린 장기 기억 노트" | 44 |
| 9 | 🧪 부록 A — 인플레이스 연산: — "시험 채점 전에 답안지를 고치면 안 된다!" — ⚠️ 인플레이스 (위험): 원래 데이터가 사라짐! | 17 |
| 10 | 🧪 부록 B — get_item 함수: "선택한 자리에만 칭찬 돌려주기" | 47 |
| 11 | ☁️ 부록 C — 구글 콜랩: "무료 GPU를 빌려 쓰는 클라우드 주방" — ① 먼저: 런타임 → 런타임 유형 변경 → 하드웨어 가속기를 GPU로 설정! | 28 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-03-dezero-framework/04-03-06-inplace-getitem-colab
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_03_06_inplace_getitem_colab.py
```

또는 `04_03_06_inplace_getitem_colab.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-03-04 의 코드 실행 결과 (`model` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'model' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 3/11 블록 성공

## 추출 시 수정 사항

- 주피터 전용 명령(`%matplotlib`, `!pip` 등)은 .py 에서 주석 처리했습니다 (노트북에는 원문 그대로).

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
