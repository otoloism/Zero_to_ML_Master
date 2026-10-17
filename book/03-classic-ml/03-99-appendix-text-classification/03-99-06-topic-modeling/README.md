# 토픽 모델링은 보통 TF-IDF보다 단순 빈도(CountVectorizer)를 사용

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-99-06 🗂 토픽 모델링과 주피터 종합 실습 — 라벨 없이 주제 찾기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_99_06_topic_modeling.py`](03_99_06_topic_modeling.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_99_06_topic_modeling.ipynb`](03_99_06_topic_modeling.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 3단계 — 코드로 직접 토픽 추출하기 | 22 |
| 2 | 4단계 — 주피터 노트북 종합 실습: 03-99 부록 전체를 잇는 엔드투엔드 파이프라인 — 03-99 부록 종합 파이프라인 | 29 |

## 실행 방법

```bash
cd book/03-classic-ml/03-99-appendix-text-classification/03-99-06-topic-modeling
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_99_06_topic_modeling.py
```

또는 `03_99_06_topic_modeling.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`, 2026-10-17 재검증)
- 필요 사항: 앞 절 03-99-01 의 코드 실행 결과 (`df` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'df' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

## 추출 시 수정 사항

- 위키독스 페이지([wikidocs.net/439687](https://wikidocs.net/439687))의 코드 상자를 그대로 옮겼습니다 — 블록 1개 = 코드 상자 1개, 순서·내용 동일 (2026-10-17 페이지 기준 정렬). 출력 결과·mermaid 다이어그램·글자 표(ASCII) 상자는 코드 상자로 세지 않았습니다.
- 예전에 넣었던 보충 import(`Block 0 [추가]`), 주석 처리, 오타 수정은 페이지와 1:1로 맞추기 위해 되돌렸습니다.

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
