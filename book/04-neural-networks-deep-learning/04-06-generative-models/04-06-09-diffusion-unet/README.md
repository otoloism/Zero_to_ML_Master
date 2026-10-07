# 04-06-09 확산 모델구현 —U-Net과 데이터 생성

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-09 확산 모델구현 —U-Net과 데이터 생성.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_09_diffusion_unet.py`](04_06_09_diffusion_unet.py) | 코드 블록 10개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_09_diffusion_unet.ipynb`](04_06_09_diffusion_unet.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 코드 구현 — 순진한 방식(느림) | 10 |
| 2 | Python — diffusion.py | 23 |
| 3 | 손실 함수 — 다시 한 번 04-01-03의 MSE | 9 |
| 4 | 코드 구현 | 25 |
| 5 | 부품 ① ResidualBlock — 시간 정보를 주입하는 기본 블록 | 26 |
| 6 | 부품 ② 전체 U-Net 조립 | 52 |
| 7 | Python — train.py | 36 |
| 8 | Python — sample.py | 25 |
| 9 | 📖 해설 | 3 |
| 10 | 📖 해설 | 12 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-09-diffusion-unet
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_09_diffusion_unet.py
```

또는 `04_06_09_diffusion_unet.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `x0` 이(가) 이 절의 뒤쪽 블록에서야 정의됩니다 (책 서술 순서상 앞 블록은 설명용 조각).
- 스크립트가 멈춘 지점: `NameError: name 'x0' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 8/10 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
