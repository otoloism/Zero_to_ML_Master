# 04-06-11 다변량MLE도출 ·옌센 부등식· 계층형VAE· 수식 기호 목록

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-11 다변량MLE도출 ·옌센 부등식· 계층형VAE· 수식 기호 목록.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_11_multivariate_mle_jensen_hvae.py`](04_06_11_multivariate_mle_jensen_hvae.py) | 코드 블록 1개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_11_multivariate_mle_jensen_hvae.ipynb`](04_06_11_multivariate_mle_jensen_hvae.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-06-11-C1 3단계 — 부록 C: 계층형 VAE (HVAE) — VAE와 확산 모델의 다리 | 30 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-11-multivariate-mle-jensen-hvae
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_11_multivariate_mle_jensen_hvae.py
```

또는 `04_06_11_multivariate_mle_jensen_hvae.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 추출 시 수정 사항

- Block 0 추가: `import torch.nn as nn` — 책에서는 앞 절에서 import 했거나 뒤 블록에서야 import 하는 모듈이라, 단독 실행을 위해 보충했습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
