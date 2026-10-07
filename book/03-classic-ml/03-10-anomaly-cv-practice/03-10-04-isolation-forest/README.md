# 03-10-04 Isolation Forest — 스무고개로 범인 찾기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-10-04 Isolation Forest — 스무고개로 범인 찾기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_10_04_isolation_forest.py`](03_10_04_isolation_forest.py) | 코드 블록 10개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_10_04_isolation_forest.ipynb`](03_10_04_isolation_forest.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | ▸ 직접 시뮬레이션해 보기 | 47 |
| 2 | ▸ 유도 — 왜 이 식인가 | 32 |
| 3 | ▸ 구현 관점 — 사이킷런은 부호를 뒤집는다 | 20 |
| 4 | ▸ 구현 관점 — 사이킷런은 부호를 뒤집는다 | 23 |
| 5 | ▸ 구현 관점 — 사이킷런은 부호를 뒤집는다 | 4 |
| 6 | ▸ 세 모델 최종 비교 | 24 |
| 7 | ▸ 결과 시각화 — 강의 자료 19p | 17 |
| 8 | ▸ ② 속도 — 이 모델이 존재하는 진짜 이유 | 33 |
| 9 | ▸ ③ 알려진 약점 — 축에 평행한 선만 긋는다 | 43 |
| 10 | ▸ 프레임워크 비교 | 38 |

## 실행 방법

```bash
cd book/03-classic-ml/03-10-anomaly-cv-practice/03-10-04-isolation-forest
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_10_04_isolation_forest.py
```

또는 `03_10_04_isolation_forest.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 03-10-01 의 코드 실행 결과 (`X_train` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'X_train' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 3/10 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
