# 04-06-07 변이형 오토인코더 (VAE) — 신경망 기반 생성 모델

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-07 변이형 오토인코더 (VAE) — 신경망 기반 생성 모델.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_07_vae.py`](04_06_07_vae.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_07_vae.ipynb`](04_06_07_vae.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 구현 관점 | 76 |
| 2 | 핵심 정의 | 11 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-07-vae
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_07_vae.py
```

또는 `04_06_07_vae.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 04-06-06 의 코드 실행 결과 (`train_loader` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'train_loader' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
