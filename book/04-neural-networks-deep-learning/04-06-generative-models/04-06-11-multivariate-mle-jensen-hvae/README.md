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

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`, 2026-10-17 재검증)
- 상세: `nn` 이(가) 이 절 안에서 정의되지 않습니다 — 앞 절에서 정의·import 했다고 가정한 페이지 원문 그대로의 코드 조각입니다.
- 스크립트가 멈춘 지점: `NameError: name 'nn' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/1 블록 성공

## 추출 시 수정 사항

- 위키독스 페이지([wikidocs.net/439809](https://wikidocs.net/439809))의 코드 상자를 그대로 옮겼습니다 — 블록 1개 = 코드 상자 1개, 순서·내용 동일 (2026-10-17 페이지 기준 정렬). 출력 결과·mermaid 다이어그램·글자 표(ASCII) 상자는 코드 상자로 세지 않았습니다.
- 예전에 넣었던 보충 import(`Block 0 [추가]`), 주석 처리, 오타 수정은 페이지와 1:1로 맞추기 위해 되돌렸습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
