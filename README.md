# Zero to 머신러닝 딥러닝 Master

이 책은 WikiDocs와 연동된 GitHub 리포지토리입니다.

## Medium 시리즈 코드

책 **[Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)** 를 Medium 시리즈로 연재하며, 각 글의 실습 코드를 [`medium-series/`](medium-series/) 폴더에 정리합니다. 폴더마다 실행 스크립트(`.py`), 실행 결과가 담긴 노트북(`.ipynb`), `requirements.txt`, 설명 `README.md` 가 있습니다.

| # | 주제 | 코드 | Medium 글 | 책 원문 |
|---|---|---|---|---|
| 03 | 벡터 — 덧셈·Norm·내적·코사인 유사도 | [03-vectors](medium-series/03-vectors/) | [읽기](https://medium.com/p/9a7f826c3b2e) | [00-01](https://wikidocs.net/439839) |
| 04 | 행렬 — shape·행렬 곱셈·역행렬 | [04-matrices](medium-series/04-matrices/) | TODO: 게시 후 URL | [00-02](https://wikidocs.net/439843) |
| 05 | 최소제곱법과 정규방정식 | [05-least-squares](medium-series/05-least-squares/) | TODO: 게시 후 URL | [00-03](https://wikidocs.net/439841) |

- 📘 책 (WikiDocs): https://wikidocs.net/book/21464
- 🏠 홈페이지: https://mldict.net
- ✍️ 기술 블로그: https://wikidocs.net/blog/@mldict/

## 사용 방법

1. `TOC.md` 파일에서 목차 구조를 정의하세요.
2. `pages/` 디렉토리에 마크다운 파일을 추가하세요.
3. 변경사항을 push하면 WikiDocs에 자동으로 반영됩니다.

## 페이지 정렬 규칙 (중요!)

WikiDocs는 페이지를 **제목 알파벳순**으로 자동 정렬합니다.
원하는 순서를 유지하려면 제목에 **번호를 붙이세요**:

```markdown
# TOC.md 예시
* [01. 시작하기](pages/01-getting-started.md)
  * [01-1. 설치](pages/01-1-install.md)
  * [01-2. 환경설정](pages/01-2-config.md)
* [02. 기본 문법](pages/02-basics.md)
* [03. 심화 학습](pages/03-advanced.md)
```

## 이미지 사용

이미지는 `assets/` 디렉토리에 저장하고 상대 경로로 참조하세요:

```markdown
![이미지 설명](./assets/example.png)
```
