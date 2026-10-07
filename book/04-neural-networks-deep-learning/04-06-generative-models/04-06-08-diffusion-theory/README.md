# 04-06-08 확산 모델이론 —VAE에서 Diffusion으로

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-08 확산 모델이론 —VAE에서 Diffusion으로.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_08_diffusion_theory.py`](04_06_08_diffusion_theory.py) | 코드 블록 3개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_08_diffusion_theory.ipynb`](04_06_08_diffusion_theory.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-06-08-02 2단계 — 순방향 과정: 이미지를 노이즈로 | 25 |
| 2 | 04-06-08-05 5단계 — DDPM 학습 루프 구현 | 34 |
| 3 | 04-06-08-05 5단계 — DDPM 학습 루프 구현 | 28 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-08-diffusion-theory
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_08_diffusion_theory.py
```

또는 `04_06_08_diffusion_theory.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
