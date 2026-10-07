# 03-02-02 경사하강법의 직관 — 학습률의 영향

> 보폭이 너무 크면 넘어지고, 너무 작으면 하염없는 산행

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-02-02 경사하강법의 직관 - 학습률의 영향.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_02_02_learning_rate_intuition.py`](03_02_02_learning_rate_intuition.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_02_02_learning_rate_intuition.ipynb`](03_02_02_learning_rate_intuition.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | learning_rate_experiment.py | 40 |
| 2 | 다. 구현 문제 | 12 |

## 실행 방법

```bash
cd book/03-classic-ml/03-02-parameter-learning/03-02-02-learning-rate-intuition
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_02_02_learning_rate_intuition.py
```

또는 `03_02_02_learning_rate_intuition.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `n_epochs` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'n_epochs' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
