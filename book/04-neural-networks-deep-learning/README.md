# 04권 신경망과 딥러닝 이론

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `04권  신경망과 딥러닝 이론.md`

## 하위 절

| 번호 | 절 | 코드 블록 | 상태 |
|---|---|---|---|
| 04-01 | [딥러닝 이론과 구현](04-01-deep-learning-theory/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-01 | [퍼셉트론— 신경망의 가장 단순한 조상](04-01-deep-learning-theory/04-01-01-perceptron/) | 3 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-02 | [신경망 —활성화 함수와 순전파](04-01-deep-learning-theory/04-01-02-activation-forward/) | 7 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-03 | [신경망 학습 —손실 함수와 경사 하강법](04-01-deep-learning-theory/04-01-03-loss-gradient-descent/) | 6 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-04 | [오차역전파법 — 계산 그래프로 이해하기](04-01-deep-learning-theory/04-01-04-backpropagation/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-05 | [학습 관련 기술들 — 옵티마이저·초기화·정규화](04-01-deep-learning-theory/04-01-05-optimizers-init-regularization/) | 7 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-06 | [합성곱 신경망 (CNN) — 이미지 인식의 혁명](04-01-deep-learning-theory/04-01-06-cnn/) | 3 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-07 | [딥러닝 — 더 깊은 네트워크와 응용](04-01-deep-learning-theory/04-01-07-deeper-networks/) | 8 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-01-08 | [Softmax-with-Loss 계층의 계산 그래프 유도](04-01-deep-learning-theory/04-01-08-softmax-with-loss/) | 9 | ✅ 실행 OK |
| 04-02 | [강화 학습 알고리즘](04-02-reinforcement-learning/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-01 | [밴디트 문제 — 강화학습의 작은 출발점](04-02-reinforcement-learning/04-02-01-bandit/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-02 | [마르코프 결정 과정(MDP) — 강화학습의 수학적 틀](04-02-reinforcement-learning/04-02-02-mdp/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-03 | [벨만 방정식— 가치 함수의 재귀 구조](04-02-reinforcement-learning/04-02-03-bellman/) | 5 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-04 | [동적 프로그래밍 (DP) — 환경 모델을 알 때](04-02-reinforcement-learning/04-02-04-dynamic-programming/) | 4 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-05 | [몬테카를로(MC) 방법 — 환경 모델 없이](04-02-reinforcement-learning/04-02-05-monte-carlo/) | 7 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-06 | [TD법 —SARSA와Q-Learning](04-02-reinforcement-learning/04-02-06-td-sarsa-qlearning/) | 4 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-07 | [신경망과 Q 러닝 — 함수 근사로 가는 길](04-02-reinforcement-learning/04-02-07-nn-q-learning/) | 2 | ❌ 실행 오류 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-08 | [DQN— 아타리를 정복한 딥 강화학습의 시작](04-02-reinforcement-learning/04-02-08-dqn/) | 3 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-09 | [정책 경사법— Policy Gradient와Actor-Critic](04-02-reinforcement-learning/04-02-09-policy-gradient-actor-critic/) | 2 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-10 | [한 걸음 더 — 최신 RL과 실세계 응용](04-02-reinforcement-learning/04-02-10-modern-rl/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-02-11 | [오프-정책 MC · n단계 TD · DoubleDQN·정책 경사법증명](04-02-reinforcement-learning/04-02-11-offpolicy-nstep-doubledqn/) | 4 | 🧩 문맥 필요(조각) |
| 04-03 | [딥러닝 프레임워크 — DeZero 직접 만들기](04-03-dezero-framework/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-03-01 | [미분자동 계산 (1 10단계) — 오토그라드의 탄생](04-03-dezero-framework/04-03-01-autograd-steps-01-10/) | 9 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-03-02 | [자연스러운 코드 (11 24단계) — 연산자 오버로딩](04-03-dezero-framework/04-03-02-operator-overloading-steps-11-24/) | 8 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-03-03 | [고차미분계산 (25 36단계) —헤시안과 뉴턴법](04-03-dezero-framework/04-03-03-higher-order-diff-steps-25-36/) | 7 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-03-04 | [신경망 만들기 (37 51단계) —PyTorchnn Module 직접 구현](04-03-dezero-framework/04-03-04-neural-net-steps-37-51/) | 14 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-03-05 | [DeZero의 도전 (52 60단계) — GPU·CNN·RNN까지](04-03-dezero-framework/04-03-05-gpu-cnn-rnn-steps-52-60/) | 10 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-03-06 | [인플레이스 연산 · get_item 함수 · 구글 콜랩](04-03-dezero-framework/04-03-06-inplace-getitem-colab/) | 11 | 🧩 문맥 필요(조각) |
| 04-04 | [순환 신경망과 자연어 처리](04-04-rnn-nlp/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-01 | [신경망 복습](04-04-rnn-nlp/04-04-01-nn-review/) | 6 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-02 | [자연어와 단어의 분산 표현 — 의미를벡터로](04-04-rnn-nlp/04-04-02-distributed-representation/) | 5 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-03 | [word2vec— 추론 기반 단어임베딩](04-04-rnn-nlp/04-04-03-word2vec/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-04 | [word2vec속도 개선 — Embedding 계층과네거티브 샘플링](04-04-rnn-nlp/04-04-04-word2vec-speedup/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-05 | [순환 신경망 (RNN) — 시간을 기억하는 신경망](04-04-rnn-nlp/04-04-05-rnn/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-06 | [게이트가 추가된RNN—LSTM과 기울기 문제 해결](04-04-rnn-nlp/04-04-06-lstm/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-07 | [RNN을 사용한 문장 생성 —seq2seq](04-04-rnn-nlp/04-04-07-seq2seq/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-08 | [어텐션—Transformer로 가는 다리](04-04-rnn-nlp/04-04-08-attention/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-09 | [시그모이드와 tanh 함수의미분유도](04-04-rnn-nlp/04-04-09-sigmoid-tanh-derivatives/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-10 | [WordNet 맛보기 — 단어의 의미 사전](04-04-rnn-nlp/04-04-10-wordnet/) | 1 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-04-11 | [GRU—LSTM의 경량 변형](04-04-rnn-nlp/04-04-11-gru/) | 1 | ✅ 실행 OK |
| 04-05 | [자연어 처리 입문 — 컴퓨터에게 말을 가르치는 첫걸음](04-05-nlp-intro/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-05-01 | [자연어 처리(NLP)란 무엇인가 — 컴퓨터에게 말을 가르치는 일](04-05-nlp-intro/04-05-01-what-is-nlp/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-05-02 | [텍스트 전처리 파이프라인 한눈에 보기 — 택배 분류장 비유](04-05-nlp-intro/04-05-02-preprocessing-pipeline-overview/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-05-03 | [소문자 변환과 특수문자·구두점 제거 — 글자 다듬기](04-05-nlp-intro/04-05-03-lowercase-punctuation/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-05-04 | [개체명 인식(NER) — 문장 속 _이름표_ 찾아내기](04-05-nlp-intro/04-05-04-ner/) | 2 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-05-05 | [품사 태깅(POS Tagging) — 각 단어의 _역할_ 알아내기](04-05-nlp-intro/04-05-05-pos-tagging/) | 2 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-05-06 | [전처리 파이프라인 종합 — 컨베이어 벨트 끝까지 완성하기](04-05-nlp-intro/04-05-06-preprocessing-pipeline/) | 1 | ⏭️ 외부 요구사항 |
| 04-06 | [이미지 생성 모델의 원리](04-06-generative-models/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-01 | [정규 분포 — 모든 생성 모델의 출발점](04-06-generative-models/04-06-01-normal-distribution/) | 3 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-02 | [최대 가능도 추정 (MLE) — 생성 모델 학습의 원리](04-06-generative-models/04-06-02-mle/) | 4 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-03 | [물고기 5마리 x (길이, 무게) 2개 특성 데이터](04-06-generative-models/04-06-03-multivariate-normal/) | 4 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-04 | [x1 방향으로 2만큼 떨어진 점 vs x2 방향으로 2만큼 떨어진 점](04-06-generative-models/04-06-04-gmm/) | 4 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-05 | [키 분포: 여성(평균160, 분산36) 60%, 남성(평균175, 분산49) 40%](04-06-generative-models/04-06-05-em-algorithm/) | 4 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-06 | [신경망 —PyTorch기초](04-06-generative-models/04-06-06-pytorch-basics/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-07 | [변이형 오토인코더 (VAE) — 신경망 기반 생성 모델](04-06-generative-models/04-06-07-vae/) | 2 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-08 | [확산 모델이론 —VAE에서 Diffusion으로](04-06-generative-models/04-06-08-diffusion-theory/) | 3 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-09 | [확산 모델구현 —U-Net과 데이터 생성](04-06-generative-models/04-06-09-diffusion-unet/) | 10 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-10 | [확산 모델응용 — Classifier-Free Guidance와Stable Diffusion](04-06-generative-models/04-06-10-cfg-stable-diffusion/) | 8 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-06-11 | [다변량MLE도출 ·옌센 부등식· 계층형VAE· 수식 기호 목록](04-06-generative-models/04-06-11-multivariate-mle-jensen-hvae/) | 1 | ✅ 실행 OK |
| 04-07 | [대규모 언어 모델(LLM) 이해하기 — GPT의 시대는 무엇이 다른가](04-07-llm/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-07-01 | [API 키는 환경변수로 등록 (자세한 설정은 04-08장에서 다룹니다)](04-07-llm/04-07-01-llm-vs-classic-lm/) | 4 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-07-02 | [우리는 GPU도, 학습 데이터도, 파라미터 튜닝도 필요 없습니다.](04-07-llm/04-07-02-why-llm-challenges/) | 2 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-07-03 | [명확한 프롬프트 (훨씬 나은 결과)](04-07-llm/04-07-03-gpt3-challenges/) | 4 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-07-04 | [비공개 API (GPT)](04-07-llm/04-07-04-llm-types/) | 1 | ⏭️ 외부 요구사항 |
| 04-08 | [RAG와 랭체인 실전 — LLM의 한계를 코드로 넘어서기](04-08-rag-langchain/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-08-01 | [OPENAI_API_KEY=sk-...](04-08-rag-langchain/04-08-01-llm-app-setup/) | 3 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-08-02 | [→ SQL 인젝션 취약점을 정확히 지적하는 응답](04-08-rag-langchain/04-08-02-prompt-engineering/) | 3 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;04-08-03 | [1) 문서 로드](04-08-rag-langchain/04-08-03-rag-langchain-intro/) | 2 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-08-04 | [병렬 실행: 요약과 번역을 동시에 수행 (순차 대비 시간 단축)](04-08-rag-langchain/04-08-04-cloud-llm-advanced-chains/) | 3 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-08-05 | [내부적으로: 1) "검색이 필요하다" 판단 → 2) web_search 호출 → 3) 결과를 바탕으로 답변 생성](04-08-rag-langchain/04-08-05-web-search-prompt-compression/) | 2 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-08-06 | [멀티 에이전트— 협력하는LLM팀 만들기](04-08-rag-langchain/04-08-06-multi-agent/) | 2 | 🧩 문맥 필요(조각) |
| 04-09 | LLM·AI가 만드는 과거·현재·미래 — 큰 그림으로 정리하기 | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-09-01 | LLM·AI 관련 주요 기술 트렌드 — 지금까지의 흐름 한눈에 | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-09-02 | 컴퓨팅 파워와 대규모 데이터셋의 미래 — 연료와 엔진 | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-09-03 | 대규모 언어 모델의 진화와 비즈니스 영향 — 산업의 재편 | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;04-09-04 | AI가 유도하는 행동 트렌드(사회적 측면) — 그리고 전체 여정 마무리 | — | 코드 없음 |

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
