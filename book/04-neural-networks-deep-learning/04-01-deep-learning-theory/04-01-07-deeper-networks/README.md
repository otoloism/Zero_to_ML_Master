# 04-01-07 딥러닝 — 더 깊은 네트워크와 응용

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-01-07 딥러닝 — 더 깊은 네트워크와 응용.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_01_07_deeper_networks.py`](04_01_07_deeper_networks.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_01_07_deeper_networks.ipynb`](04_01_07_deeper_networks.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 친절하게 풀어보기 — numpy를 불러옵니다. "np"라는 짧은 별명을 붙여주는 것이 관례입니다. | 47 |
| 2 | 04-01-07-A 28×28 그림을 한 줄로 쭉 펼치면 무슨 일이 벌어질까요? | 19 |
| 3 | 04-01-07-C 숫자로 하나하나 직접 계산해 보기 — numpy를 불러옵니다. | 42 |
| 4 | 04-01-07-D 여러 종류의 필터로 여러 가지 특징 찾기 | 42 |
| 5 | 04-01-07-F 패딩(Padding) — 그림 가장자리에 액자 테두리 둘러주기 | 33 |
| 6 | 결과 크기 = (H − F + 2P) ÷ S + 1 — 결과 크기 = (H - F + 2P) / S + 1 | 18 |
| 7 | 04-01-07-I 맥스 풀링 — 한 구역에서 가장 힘센 신호만 남기기 | 25 |
| 8 | 04-01-07-K 핵심: 도장이 지나갈 자리들을 미리 한 줄씩 펼쳐서 쌓아두기 | 37 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-01-deep-learning-theory/04-01-07-deeper-networks
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_01_07_deeper_networks.py
```

또는 `04_01_07_deeper_networks.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
