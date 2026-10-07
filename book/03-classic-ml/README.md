# 03권. 초기(전통적) 머신러닝으로 기초 다지기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03권  초기(전통적) 머신러닝으로 기초 다지기.md`

## 하위 절

| 번호 | 절 | 코드 블록 | 상태 |
|---|---|---|---|
| 03-01 | [머신러닝 기초](03-01-ml-basics/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-01-01 | 머신러닝이란 무엇인가? | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;03-01-02 | 회귀 vs 분류 | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;03-01-03 | 비지도학습 — 군집화와 비군집화 | — | 코드 없음 |
| &nbsp;&nbsp;&nbsp;&nbsp;03-01-04 | [선형회귀분석 개요](03-01-ml-basics/03-01-04-linear-regression-overview/) | 2 | ✅ 실행 OK |
| 03-02 | [Parameter Learning](03-02-parameter-learning/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-02-01 | [경사하강법 개념과 절차](03-02-parameter-learning/03-02-01-gradient-descent-concept/) | 3 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-02-02 | [경사하강법의 직관 — 학습률의 영향](03-02-parameter-learning/03-02-02-learning-rate-intuition/) | 2 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-02-03 | [선형회귀를 위한 경사하강법 — 수식 유도](03-02-parameter-learning/03-02-03-gd-for-linear-regression/) | 5 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-02-04 | [NumPy/PyTorch 오토그라드·옵티마이저 실습](03-02-parameter-learning/03-02-04-numpy-pytorch-autograd/) | 14 | 🧩 문맥 필요(조각) |
| 03-03 | [Feature 설계와 분류 모델](03-03-features-classification/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-03-01 | [특성 설계와 다항 회귀](03-03-features-classification/03-03-01-features-polynomial-regression/) | 3 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-03-02 | [정규 방정식](03-03-features-classification/03-03-02-normal-equation/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-03-03 | [로지스틱 회귀 — 가설의 표현](03-03-features-classification/03-03-03-logistic-regression-hypothesis/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-03-04 | [결정 경계](03-03-features-classification/03-03-04-decision-boundary/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-03-05 | [비용 함수·경사하강법·다중 클래스 분류](03-03-features-classification/03-03-05-logistic-cost-gd-multiclass/) | 3 | ✅ 실행 OK |
| 03-04 | [정규화와 신경망 기초](03-04-regularization-nn-basics/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-04-01 | [과대적합 문제와 정규화의 아이디어](03-04-regularization-nn-basics/03-04-01-overfitting-regularization/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-04-02 | [정규화된 선형회귀](03-04-regularization-nn-basics/03-04-02-regularized-linear-regression/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-04-03 | [정규화된 로지스틱 회귀](03-04-regularization-nn-basics/03-04-03-regularized-logistic-regression/) | 3 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-04-04 | [신경망의 등장과 모델 표현](03-04-regularization-nn-basics/03-04-04-neural-network-representation/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-04-05 | [신경망으로 논리 게이트 만들기](03-04-regularization-nn-basics/03-04-05-nn-logic-gates/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-04-06 | [신경망 비용 함수와 역전파](03-04-regularization-nn-basics/03-04-06-nn-cost-backprop/) | 2 | ✅ 실행 OK |
| 03-05 | [1단계: 데이터 생성](03-05-clustering-dim-reduction/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-01 | [K-평균 알고리즘 개념과 절차](03-05-clustering-dim-reduction/03-05-01-kmeans-concept/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-02 | [최적화 목표 — 왜곡 비용함수](03-05-clustering-dim-reduction/03-05-02-distortion-cost/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-03 | [무작위 초기화와 지역 최적해](03-05-clustering-dim-reduction/03-05-03-random-init-local-optima/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-04 | [클러스터 개수 $K$ 결정하기](03-05-clustering-dim-reduction/03-05-04-choosing-k/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-05 | [K-평균 실습 — 붓꽃과 원형군집](03-05-clustering-dim-reduction/03-05-05-kmeans-iris-circles/) | 9 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-06 | [차원 축소와 PCA 문제 정의](03-05-clustering-dim-reduction/03-05-06-pca-problem/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-07 | [PCA 알고리즘 — 전처리부터 복원까지](03-05-clustering-dim-reduction/03-05-07-pca-algorithm/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-05-08 | [PCA 적용과 붓꽃 실습](03-05-clustering-dim-reduction/03-05-08-pca-iris/) | 8 | 🧩 문맥 필요(조각) |
| 03-06 | [이상탐지 이론](03-06-anomaly-detection-theory/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-06-01 | [이상탐지와 밀도추정 개념](03-06-anomaly-detection-theory/03-06-01-density-estimation/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-06-02 | [가우시안 분포와 파라미터 추정](03-06-anomaly-detection-theory/03-06-02-gaussian-parameter-estimation/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-06-03 | [이상탐지 알고리즘](03-06-anomaly-detection-theory/03-06-03-anomaly-detection-algorithm/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-06-04 | [이상탐지 시스템의 구현과 평가](03-06-anomaly-detection-theory/03-06-04-anomaly-system-evaluation/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-06-05 | [이상탐지 vs 지도학습](03-06-anomaly-detection-theory/03-06-05-anomaly-vs-supervised/) | 1 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-06-06 | [이상탐지에서의 Feature 선택](03-06-anomaly-detection-theory/03-06-06-anomaly-feature-selection/) | 2 | ✅ 실행 OK |
| 03-07 | [추천 시스템 (Recommender Systems)](03-07-recommender-systems/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-07-01 | [추천시스템 문제 정의 (Recommender System · Predicting Movie Ratings)](03-07-recommender-systems/03-07-01-recsys-problem/) | 2 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-07-02 | [콘텐츠 기반 추천 시스템 (Content Based Recommendation)](03-07-recommender-systems/03-07-02-content-based-recsys/) | 8 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-07-03 | [협업 필터링 (Collaborative Filtering)](03-07-recommender-systems/03-07-03-collaborative-filtering/) | 7 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-07-04 | [저차원 행렬분해와 평균 정규화 (Low Rank Matrix Factorization & Mean Normalization)](03-07-recommender-systems/03-07-04-low-rank-mean-normalization/) | 7 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-07-05 | [콘텐츠 기반 추천 시스템 실습 (Content Based Recommendation with TMDB 5000)](03-07-recommender-systems/03-07-05-content-based-practice/) | 17 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;03-07-06 | [협업 필터링 실습 — SGD 행렬 분해 (Matrix Factorization with Stochastic Gradient …](03-07-recommender-systems/03-07-06-cf-sgd-matrix-factorization/) | 15 | ⏭️ 외부 요구사항 |
| 03-08 | [정규화 심화 및 대규모 학습 (Regularization & Large Scale Machine Learning)](03-08-advanced-regularization-large-scale/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-08-01 | [L1 노름과 L2 노름 (L1 Norm & L2 Norm)](03-08-advanced-regularization-large-scale/03-08-01-l1-l2-norms/) | 5 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-08-02 | [L1 정규화(Lasso)와 L2 정규화(Ridge) (L1 & L2 Regularization)](03-08-advanced-regularization-large-scale/03-08-02-lasso-ridge/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-08-03 | [대규모 데이터와 학습 곡선 (Learning with Large Datasets & Learning Curve)](03-08-advanced-regularization-large-scale/03-08-03-large-data-learning-curves/) | 4 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-08-04 | [배치 · 확률적 · 미니배치 경사하강법 (Batch / Stochastic / Mini-batch Gradient Desce…](03-08-advanced-regularization-large-scale/03-08-04-batch-sgd-minibatch/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-08-05 | [SGD 수렴 진단 · 온라인 학습 · 맵리듀스 (SGD Convergence · Online Learning · Map Re…](03-08-advanced-regularization-large-scale/03-08-05-sgd-convergence-online-mapreduce/) | 6 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-08-06 | [Ridge · Lasso · ElasticNet 실습 (Regularization Practice)](03-08-advanced-regularization-large-scale/03-08-06-ridge-lasso-elasticnet-practice/) | 18 | ✅ 실행 OK |
| 03-09 | [부스팅 알고리즘](03-09-boosting/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-09-01 | [부스팅 알고리즘 부스팅의 개념과 배깅과의 차이](03-09-boosting/03-09-01-boosting-vs-bagging/) | 5 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-09-02 | [부스팅 알고리즘 AdaBoost — 적응형 부스팅](03-09-boosting/03-09-02-adaboost/) | 5 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-09-03 | [그레이디언트 부스팅 — 잔차를 학습하는 릴레이](03-09-boosting/03-09-03-gradient-boosting/) | 7 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-09-04 | [GBM의 규제 — 학습률·조기 종료·확률적·히스토그램 부스팅](03-09-boosting/03-09-04-gbm-regularization/) | 11 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-09-05 | [XGBoost — 복잡도까지 값을 매기는 부스팅](03-09-boosting/03-09-05-xgboost/) | 16 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-09-06 | [LightGBM과 Santander 종합 실습](03-09-boosting/03-09-06-lightgbm-santander/) | 15 | ✅ 실행 OK |
| 03-10 | [이상탐지·교차검증 실습](03-10-anomaly-cv-practice/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-01 | [노벨티 탐지와 아웃라이어 탐지](03-10-anomaly-cv-practice/03-10-01-novelty-outlier-detection/) | 9 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-02 | [EllipticEnvelope — 타원 울타리](03-10-anomaly-cv-practice/03-10-02-elliptic-envelope/) | 11 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-03 | [그림을 그리려면 아래 코드를 실행하세요](03-10-anomaly-cv-practice/03-10-03-local-outlier-factor/) | 9 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-04 | [Isolation Forest — 스무고개로 범인 찾기](03-10-anomaly-cv-practice/03-10-04-isolation-forest/) | 10 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-05 | [교차 검증:1](03-10-anomaly-cv-practice/03-10-05-kfold-stratified/) | 14 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-06 | [cross_val_score부터 LOOCV까지](03-10-anomaly-cv-practice/03-10-06-cross-val-score-loocv/) | 19 | ✅ 실행 OK |
| &nbsp;&nbsp;&nbsp;&nbsp;03-10-07 | [혼동행렬과 ROC-AUC](03-10-anomaly-cv-practice/03-10-07-confusion-matrix-roc-auc/) | 14 | ✅ 실행 OK |
| 03-99 | [부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들](03-99-appendix-text-classification/) | — | (하위 절 참고) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-99-01 | [positive 800](03-99-appendix-text-classification/03-99-01-eda-data-split/) | 2 | ⏭️ 외부 요구사항 |
| &nbsp;&nbsp;&nbsp;&nbsp;03-99-02 | [kernel='linear': 직선(평면) 경계 사용 (텍스트처럼 고차원엔 보통 이걸로 충분)](03-99-appendix-text-classification/03-99-02-classifier-toolbox/) | 6 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-99-03 | [depth 후보를 하나씩 바꿔가며 학습/검증 정확도를 비교합니다](03-99-appendix-text-classification/03-99-03-fitting-hyperparameter-tuning/) | 2 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-99-04 | [서로 다른 알고리즘 세 개를 준비합니다 (서로 다른 "위원"들)](03-99-appendix-text-classification/03-99-04-ensemble-imbalance-correlation/) | 4 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-99-05 | [max_features: 상위 5000개 단어만 사용 (메모리 절약)](03-99-appendix-text-classification/03-99-05-tfidf-word2vec/) | 2 | 🧩 문맥 필요(조각) |
| &nbsp;&nbsp;&nbsp;&nbsp;03-99-06 | [토픽 모델링은 보통 TF-IDF보다 단순 빈도(CountVectorizer)를 사용](03-99-appendix-text-classification/03-99-06-topic-modeling/) | 2 | 🧩 문맥 필요(조각) |

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
