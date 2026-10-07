# 04-03-02 자연스러운 코드 (11 24단계) — 연산자 오버로딩

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-03-02 자연스러운 코드 (11 24단계) — 연산자 오버로딩.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_03_02_operator_overloading_steps_11_24.py`](04_03_02_operator_overloading_steps_11_24.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_03_02_operator_overloading_steps_11_24.ipynb`](04_03_02_operator_overloading_steps_11_24.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 11~12단계 — 가변 길이 인수: 입력이 여러 개인 함수 | 24 |
| 2 | 13~14단계 — 같은 변수 반복 사용: 기울기 누적 — backward() 내부의 핵심 변경 한 줄: | 12 |
| 3 | 15~16단계 — 복잡한 그래프와 세대(generation) 정렬 — Function.__call__에서 세대 계산 | 11 |
| 4 | 17~18단계 — 메모리 관리: weakref와 retain_grad | 14 |
| 5 | 19~20단계 — 변수 사용성: shape, len, print | 21 |
| 6 | 21~22단계 — 연산자 오버로딩: +, * 를 자연스럽게! | 23 |
| 7 | 23단계 — 패키지 설계: 모듈 분리 — 디렉토리 구조: | 9 |
| 8 | 24단계 — 복잡한 함수 미분: 최적화 벤치마크로 검증 — ① Sphere 함수: z = x² + y² (가장 단순한 볼록함수) | 37 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-03-dezero-framework/04-03-02-operator-overloading-steps-11-24
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_03_02_operator_overloading_steps_11_24.py
```

또는 `04_03_02_operator_overloading_steps_11_24.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-03-01 의 코드 실행 결과 (`Variable` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'Variable' is not defined. Did you mean: 'callable'?`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/8 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import numpy as np` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
