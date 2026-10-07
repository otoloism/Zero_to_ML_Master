# 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-05 EM 알고리즘— 잠재 변수 학습의 일반 도구.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_05_em_algorithm.py`](04_06_05_em_algorithm.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_05_em_algorithm.ipynb`](04_06_05_em_algorithm.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 상세 풀이 | 22 |
| 2 | 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40% | 13 |
| 3 | 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40% | 67 |
| 4 | 키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40% | 16 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-05-em-algorithm
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_05_em_algorithm.py
```

또는 `04_06_05_em_algorithm.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-06-04 의 코드 실행 결과 (`mvn_pdf` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'mvn_pdf' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 2/4 블록 성공

## 추출 시 수정 사항

- Block 0 추가: `import numpy as np` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
