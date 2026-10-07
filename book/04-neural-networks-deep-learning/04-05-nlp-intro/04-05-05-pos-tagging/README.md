# 04-05-05 품사 태깅(POS Tagging) — 각 단어의 _역할_ 알아내기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04-05-05 품사 태깅(POS Tagging) — 각 단어의 _역할_ 알아내기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`04_05_05_pos_tagging.py`](04_05_05_pos_tagging.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`04_05_05_pos_tagging.ipynb`](04_05_05_pos_tagging.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 04-05-05-A 영어 품사 태깅 (NLTK) | 15 |
| 2 | 04-05-05-B 한국어 품사 태깅 + 명사 추출 (KoNLPy) | 16 |

## 실행 방법

```bash
cd book/04-neural-networks-deep-learning/04-05-nlp-intro/04-05-05-pos-tagging
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 04_05_05_pos_tagging.py
```

또는 `04_05_05_pos_tagging.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: NLTK 데이터 다운로드 (`nltk.download(...)`)
- 스크립트가 멈춘 지점: `LookupError: `
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
