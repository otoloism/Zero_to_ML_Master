# #31 파이썬 함수 안에서 바꾼 변수가 밖에서는 그대로인 이유 — 스코프·뮤터블·참조 카운트·GIL·클로저까지

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/31-functions-memory/31_functions_memory.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #31의 실습 코드입니다.
> 원문: 책 **02-04 「🧠 4부 — 함수와 메모리: 파이썬의 심장」**

함수·docstring·기본값·`*args`/`**kwargs`부터 지역·전역 스코프와 `global`, UnboundLocalError, 재할당 vs 내용 수정(객체 참조 전달), `==` vs `is`, 참조 카운트, 일급 함수·클로저·람다·데코레이터까지 책의 코드 블록 12개를 순서대로 실행합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [02-04 함수와 메모리: 파이썬의 심장](https://wikidocs.net/book/21464) (책 목차의 02-04) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`31_functions_memory.py`](31_functions_memory.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (`# %% [Block n]` 구분) |
| [`31_functions_memory.ipynb`](31_functions_memory.ipynb) | 같은 코드를 블록당 셀 1개로 담은 Jupyter 노트북 |
| [`requirements.txt`](requirements.txt) | 추가 패키지 없음 (파이썬 표준 기능만 사용) |

## 🧭 코드 순서

1. 02-04-M 함수 — M-1 기본 · docstring · 기본값 · 키워드 인자
2. 02-04-M 함수 — M-2 `*args` / `**kwargs` (봉지와 이름표 상자)
3. 02-04-M 함수 — M-3 재사용 가능한 입력 함수 `get_input_user` (키보드 대신 `"abc"` → `"15"` 입력 데모 포함)
4. 02-04-N 스코프 — N-1 함수 안에서 바꿔도 밖은 그대로?
5. 02-04-N 스코프 — N-2 `global` 키워드와 `b.extend()`
6. 02-04-N 스코프 — N-3 UnboundLocalError (의도된 오류를 `try/except`로 받아 메시지만 출력)
7. 02-04-O 뮤터블 vs 임뮤터블 — O-1 함수에 넘기면 어떻게 되나 (재할당 vs 내용 수정)
8. 02-04-O 뮤터블 vs 임뮤터블 — O-2 `==` vs `is`
9. 02-04-P 참조 카운트 — P-1 `sys.getrefcount`로 직접 확인
10. 02-04-Q 일급 함수 — Q-1 함수를 변수·딕셔너리에 담기
11. 02-04-Q 클로저 — Q-2 `make_adder`와 `__closure__`
12. 02-04-Q 람다 — Q-3 람다 · `map` · `filter` · 데코레이터

## ▶️ 실행 방법

```bash
python 31_functions_memory.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

> 참고: `sys.getrefcount` 결과(2 → 3 → 2)는 파이썬 버전에 따라 숫자가 조금 다를 수 있습니다. 늘었다 줄어드는 흐름만 확인하면 됩니다.

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
