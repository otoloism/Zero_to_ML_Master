# 04-06-02 최대 가능도 추정 (MLE) — 생성 모델 학습의 원리

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-02 최대 가능도 추정 (MLE) — 생성 모델 학습의 원리.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_02_mle.py`](04_06_02_mle.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_02_mle.ipynb`](04_06_02_mle.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 직관 | 12 |
| 2 | 🖼️ 시각화 — 정규분포 PDF 곡선 | 16 |
| 3 | 📐 공식 | 7 |
| 4 | 구현 관점 — 경사 상승(Gradient Ascent) | 24 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-02-mle
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_02_mle.py
```

또는 `04_06_02_mle.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
