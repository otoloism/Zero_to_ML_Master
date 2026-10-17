# 04-06-10 확산 모델응용 — Classifier-Free Guidance와Stable Diffusion

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-06-10 확산 모델응용 — Classifier-Free Guidance와Stable Diffusion.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_06_10_cfg_stable_diffusion.py`](04_06_10_cfg_stable_diffusion.py) | 코드 블록 8개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_06_10_cfg_stable_diffusion.ipynb`](04_06_10_cfg_stable_diffusion.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 구현 관점 — MNIST 라벨 조건 (방법 ①) | 23 |
| 2 | 구현 연결 — 점수를 노이즈 언어로 번역 | 11 |
| 3 | 구현 관점 ① — 학습 : 라벨 드롭아웃 | 17 |
| 4 | 구현 관점 ② — 샘플링 : 배치를 두 배로 | 32 |
| 5 | 시각화 ③ — 픽셀 확산 vs 잠재 확산 — 학습 전처리: 이미지를 잠재로 | 11 |
| 6 | 공식과 변수 정의 | 32 |
| 7 | 전체 코드 — 20줄로 보는 Stable Diffusion | 30 |
| 8 | 📖 해설 | 3 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-06-generative-models/04-06-10-cfg-stable-diffusion
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_06_10_cfg_stable_diffusion.py
```

또는 `04_06_10_cfg_stable_diffusion.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`, 2026-10-17 재검증)
- 상세: `vae` 이(가) 이 절 안에서 정의되지 않습니다 — 데이터·변수·클래스를 미리 준비했다고 가정한 설명용 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'nn' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 2/8 블록 성공

## 추출 시 수정 사항

- 위키독스 페이지([wikidocs.net/439803](https://wikidocs.net/439803))의 코드 상자를 그대로 옮겼습니다 — 블록 1개 = 코드 상자 1개, 순서·내용 동일 (2026-10-17 페이지 기준 정렬). 출력 결과·mermaid 다이어그램·글자 표(ASCII) 상자는 코드 상자로 세지 않았습니다.
- 예전에 넣었던 보충 import(`Block 0 [추가]`), 주석 처리, 오타 수정은 페이지와 1:1로 맞추기 위해 되돌렸습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
