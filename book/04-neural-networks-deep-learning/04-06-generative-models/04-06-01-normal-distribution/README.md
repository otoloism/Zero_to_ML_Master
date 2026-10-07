# 04-06-01 정규 분포 — 모든 생성 모델의 출발점

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-01 정규 분포 — 모든 생성 모델의 출발점.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_01_normal_distribution.py`](04_06_01_normal_distribution.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_01_normal_distribution.ipynb`](04_06_01_normal_distribution.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 🟢 1단계 — 확률변수와 확률분포란? | 8 |
| 2 | X̄n = (1/n) Σ Xi →n→∞ N(μ, σ²/n) | 12 |
| 3 | X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²) (독립) ⟹ X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²) | 11 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-01-normal-distribution
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_01_normal_distribution.py
```

또는 `04_06_01_normal_distribution.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
