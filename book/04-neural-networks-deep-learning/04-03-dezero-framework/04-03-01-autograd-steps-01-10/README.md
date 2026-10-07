# 04-03-01 미분자동 계산 (1 10단계) — 오토그라드의 탄생

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-03-01 미분자동 계산 (1 10단계) — 오토그라드의 탄생.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_03_01_autograd_steps_01_10.py`](04_03_01_autograd_steps_01_10.py) | 코드 블록 9개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_03_01_autograd_steps_01_10.ipynb`](04_03_01_autograd_steps_01_10.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 1단계 — 상자로서의 변수: Variable 클래스 | 11 |
| 2 | 2단계 — 변수를 낳는 함수: Function 클래스 | 24 |
| 3 | 3단계 — 함수 연결: 체인처럼 이어서 호출 | 13 |
| 4 | 4단계 — 수치 미분: "정답지"를 미리 만들기 | 17 |
| 5 | 6단계 — 수동 역전파: backward()를 직접 추가하고 하나씩 호출 | 25 |
| 6 | 7단계 — 역전파 자동화: backward() 메서드 (★ 핵심!) | 27 |
| 7 | 8단계 — 재귀에서 반복문으로 — ❌ 재귀 방식 (위험): 깊은 그래프에서 스택 오버플로! | 10 |
| 8 | 9단계 — 함수를 더 편리하게: 파이썬 함수로 감싸기 | 12 |
| 9 | 10단계 — 테스트: unittest로 자동 미분 검증 | 31 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-03-dezero-framework/04-03-01-autograd-steps-01-10
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_03_01_autograd_steps_01_10.py
```

또는 `04_03_01_autograd_steps_01_10.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
