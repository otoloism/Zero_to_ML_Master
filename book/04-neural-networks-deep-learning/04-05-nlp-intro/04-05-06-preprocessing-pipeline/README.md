# 04-05-06 전처리 파이프라인 종합 — 컨베이어 벨트 끝까지 완성하기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-05-06 전처리 파이프라인 종합 — 컨베이어 벨트 끝까지 완성하기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_05_06_preprocessing_pipeline.py`](04_05_06_preprocessing_pipeline.py) | 코드 블록 1개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_05_06_preprocessing_pipeline.ipynb`](04_05_06_preprocessing_pipeline.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-05-06-A 종합 preprocess() 함수 구현 | 35 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-05-nlp-intro/04-05-06-preprocessing-pipeline
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_05_06_preprocessing_pipeline.py
```

또는 `04_05_06_preprocessing_pipeline.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: KoNLPy (Java 필요) — 검증 환경에 설치하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'konlpy'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/1 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
