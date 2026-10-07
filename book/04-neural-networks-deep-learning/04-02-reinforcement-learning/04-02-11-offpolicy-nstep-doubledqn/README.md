# 04-02-11 오프-정책 MC · n단계 TD · DoubleDQN·정책 경사법증명

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-11 오프-정책 MC · n단계 TD · DoubleDQN·정책 경사법증명.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_11_offpolicy_nstep_doubledqn.py`](04_02_11_offpolicy_nstep_doubledqn.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_11_offpolicy_nstep_doubledqn.ipynb`](04_02_11_offpolicy_nstep_doubledqn.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-11-A 부록 A — 오프-정책 MC와 중요도 샘플링의 분산 문제 | 16 |
| 2 | 04-02-11-B 부록 B — n단계 TD: MC와 TD 사이의 모든 중간 단계 | 17 |
| 3 | 04-02-11-C 부록 C — Double DQN: max의 과대평가 편향 해결 — 기존 DQN (04-02-08) | 15 |
| 4 | 04-02-11-D 부록 D — 정책 경사 정리: 로그 미분 트릭의 수학적 유도 | 24 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-11-offpolicy-nstep-doubledqn
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_11_offpolicy_nstep_doubledqn.py
```

또는 `04_02_11_offpolicy_nstep_doubledqn.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `qnet_target` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'qnet_target' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 3/4 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
