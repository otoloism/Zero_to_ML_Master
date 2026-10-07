# 03-08-05 SGD 수렴 진단 · 온라인 학습 · 맵리듀스 (SGD Convergence · Online Learning · Map Reduce)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-08-05 SGD 수렴 진단 · 온라인 학습 · 맵리듀스.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_08_05_sgd_convergence_online_mapreduce.py`](03_08_05_sgd_convergence_online_mapreduce.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_08_05_sgd_convergence_online_mapreduce.ipynb`](03_08_05_sgd_convergence_online_mapreduce.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 4단원 — 실제로 그려서 확인하기 | 43 |
| 2 | 결과 해석 — 창(window) 크기를 바꿔 같은 학습을 다르게 바라보기 | 12 |
| 3 | 일반 로지스틱 회귀와 무엇이 다른가 | 43 |
| 4 | 왜 이것이 가능한가 — 덧셈의 성질 | 30 |
| 5 | 실습 — 실습 2 정답 — 학습률이 '변화 대응 속도'를 어떻게 바꾸는가 | 27 |
| 6 | 실습 — 실습 3 정답 — 몇 대로 쪼개든 결과는 같은가? | 22 |

## 실행 방법

```bash
cd book/03-classic-ml/03-08-advanced-regularization-large-scale/03-08-05-sgd-convergence-online-mapreduce
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_08_05_sgd_convergence_online_mapreduce.py
```

또는 `03_08_05_sgd_convergence_online_mapreduce.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
