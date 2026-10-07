# 04-02-08 DQN— 아타리를 정복한 딥 강화학습의 시작

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-08 DQN— 아타리를 정복한 딥 강화학습의 시작.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_08_dqn.py`](04_02_08_dqn.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_08_dqn.ipynb`](04_02_08_dqn.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-08-B 트릭 1: 경험 재생(Experience Replay) — "카드를 섞어서 읽기" | 29 |
| 2 | 04-02-08-C 트릭 2: 타깃 신경망(Target Network) — "과녁을 잠시 고정" | 30 |
| 3 | 04-02-08-D OpenAI Gym — 표준화된 실험 환경 | 14 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-08-dqn
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_08_dqn.py
```

또는 `04_02_08_dqn.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: gym (OpenAI Gym) — 검증 환경에 설치하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'gym'`
- 블록별 실행(오류가 나도 다음 블록 계속): 2/3 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
