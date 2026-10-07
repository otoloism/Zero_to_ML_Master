# 04-02-09 정책 경사법— Policy Gradient와Actor-Critic

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-02-09 정책 경사법— Policy Gradient와Actor-Critic.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_02_09_policy_gradient_actor_critic.py`](04_02_09_policy_gradient_actor_critic.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_02_09_policy_gradient_actor_critic.ipynb`](04_02_09_policy_gradient_actor_critic.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-02-09-B REINFORCE — "-log(π) × G"가 교차 엔트로피의 변형 | 24 |
| 2 | 04-02-09-C Actor-Critic — "배우(Actor)와 평론가(Critic)" — Actor 업데이트: G 대신 어드밴티지(G - V(s)) 사용 | 10 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-02-reinforcement-learning/04-02-09-policy-gradient-actor-critic
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_02_09_policy_gradient_actor_critic.py
```

또는 `04_02_09_policy_gradient_actor_critic.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `Model` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'Model' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
