# 04-05-04 개체명 인식(NER) — 문장 속 _이름표_ 찾아내기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-05-04 개체명 인식(NER) — 문장 속 _이름표_ 찾아내기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_05_04_ner.py`](04_05_04_ner.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_05_04_ner.ipynb`](04_05_04_ner.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-05-04-A 영어 NER — spaCy로 직접 해보기 — pip install spacy && python -m spacy download en_core… | 10 |
| 2 | 04-05-04-B 한국어 NER — Transformers 파이프라인 — pip install transformers torch | 14 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-05-nlp-intro/04-05-04-ner
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_05_04_ner.py
```

또는 `04_05_04_ner.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: spaCy + 언어 모델 — 검증 환경에 설치하지 않음
- 스크립트가 멈춘 지점: `ModuleNotFoundError: No module named 'spacy'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
