# -*- coding: utf-8 -*-
"""
01-02🔤그리스 문자와 수학 기호 — ML 수식의 알파벳 (예제 100선)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-02🔤그리스 문자와 수학 기호 — ML 수식의 알파벳 (예제 100선).md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 0] [추가] 책에서는 앞 절에서 import 한 모듈 — 단독 실행을 위해 보충 (원문에는 없음)
import numpy as np
import torch


# %% [Block 1] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
theta = theta - alpha * grad          # NumPy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.step()                      # PyTorch


# %% [Block 2] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss.backward()        # 역전파로 모든 파라미터의 편미분 계산
g = param.grad         # 계산된 기울기 꺼내기


# %% [Block 3] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
m = beta1 * m + (1 - beta1) * g       # 지수이동평균
torch.optim.Adam(params, betas=(0.9, 0.999))


# %% [Block 4] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
v = beta2 * v + (1 - beta2) * g ** 2
update = g / (np.sqrt(v) + 1e-8)      # 많이 움직인 축은 보폭 축소


# %% [Block 5] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)


# %% [Block 6] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
theta -= alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)


# %% [Block 7] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
# ⚠️ 이 블록은 앞뒤 문맥 없이 단독으로는 문법 오류가 나는 코드 조각이라 주석 처리했습니다 (노트북에는 원문 그대로).
# if np.linalg.norm(grad) < 1e-5:
#     break                              # 수렴했으므로 학습 종료


# %% [Block 8] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
sched.step()


# %% [Block 9] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)


# %% [Block 10] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
H = np.array([[2.0, 0.0], [0.0, 4.0]])
theta = theta - np.linalg.inv(H) @ grad


# %% [Block 11] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
mse = np.mean((y - y_pred) ** 2)
loss = torch.nn.MSELoss()(pred, target)


# %% [Block 12] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
mae = np.mean(np.abs(y - y_pred))
loss = torch.nn.L1Loss()(pred, target)


# %% [Block 13] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = torch.nn.HuberLoss(delta=1.0)(pred, target)
from sklearn.linear_model import HuberRegressor


# %% [Block 14] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = torch.nn.BCELoss()(pred, target)
loss = torch.nn.BCEWithLogitsLoss()(logit, target)   # 수치적으로 안전


# %% [Block 15] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = torch.nn.CrossEntropyLoss()(logits, labels)   # softmax 내장
loss = -np.sum(y_onehot * np.log(y_pred + 1e-12))


# %% [Block 16] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = np.maximum(0, 1 - y * f_x)
from sklearn.svm import LinearSVC   # loss='hinge'


# %% [Block 17] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
pt = torch.exp(-ce_loss)
focal = alpha * (1 - pt) ** 2 * ce_loss


# %% [Block 18] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
kl = np.sum(p * np.log(p / q))
kl = torch.nn.KLDivLoss(reduction='batchmean')(q.log(), p)


# %% [Block 19] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = torch.nn.TripletMarginLoss(margin=1.0)(anchor, pos, neg)


# %% [Block 20] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
sim = torch.nn.functional.cosine_similarity(a, b, dim=0)


# %% [Block 21] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X, y)
posterior = model.predict_proba(X_new)


# %% [Block 22] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
ex = np.sum(x * p)
ex = np.mean(samples)     # 표본으로 근사(몬테카를로)


# %% [Block 23] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
var = np.mean(x ** 2) - np.mean(x) ** 2
var = np.var(x)


# %% [Block 24] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
sigma = np.std(x)
z = (x - np.mean(x)) / np.std(x)   # 표준화


# %% [Block 25] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
cov = np.cov(x, y)[0, 1]
C = np.cov(X.T)          # 특성 간 공분산 행렬


# %% [Block 26] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
rho = np.corrcoef(x, y)[0, 1]
import pandas as pd
df.corr()


# %% [Block 27] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from scipy.stats import norm
pdf = norm.pdf(x, loc=mu, scale=sigma)
samples = np.random.normal(mu, sigma, size=1000)


# %% [Block 28] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
nll = -np.sum(np.log(likelihood))   # 최소화 문제로 변환
res = scipy.optimize.minimize(nll_fn, theta0)


# %% [Block 29] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
log_lik = np.sum(np.log(probs + 1e-12))
loss = -torch.log_softmax(logits, dim=-1).gather(1, y).sum()


# %% [Block 30] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from scipy.stats import binom
prob = binom.pmf(k=3, n=10, p=0.5)
mask = np.random.binomial(1, 0.5, size=shape)


# %% [Block 31] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
z = W @ x + b
layer = torch.nn.Linear(in_features=784, out_features=128)


# %% [Block 32] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
s = 1 / (1 + np.exp(-x))
s = torch.sigmoid(x)


# %% [Block 33] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
t = np.tanh(x)
t = torch.tanh(x)


# %% [Block 34] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
r = np.maximum(0, x)
r = torch.relu(x)


# %% [Block 35] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
r = np.where(x > 0, x, 0.01 * x)
r = torch.nn.LeakyReLU(0.01)(x)


# %% [Block 36] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
e = np.exp(x - np.max(x))    # 오버플로 방지
p = e / e.sum()
p = torch.softmax(x, dim=-1)


# %% [Block 37] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
g = torch.nn.functional.gelu(x)
g = 0.5 * x * (1 + torch.erf(x / np.sqrt(2)))


# %% [Block 38] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
s = x * torch.sigmoid(x)
s = torch.nn.SiLU()(x)


# %% [Block 39] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
dW = dz @ x.T          # 수동 역전파
loss.backward()        # 자동 미분이 연쇄법칙 수행


# %% [Block 40] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
pred = np.argmax(probs, axis=1)
pred = logits.argmax(dim=-1)


# %% [Block 41] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
x_hat = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)
bn = torch.nn.BatchNorm2d(num_features=64)


# %% [Block 42] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
y = gamma * x_hat + beta
bn.weight   # 감마,  bn.bias  # 베타


# %% [Block 43] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
ln = torch.nn.LayerNorm(normalized_shape=512)
out = ln(x)


# %% [Block 44] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
torch.optim.Adam(params, weight_decay=1e-4)
from sklearn.linear_model import Ridge   # alpha=lambda


# %% [Block 45] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
l1 = lam * sum(p.abs().sum() for p in model.parameters())
from sklearn.linear_model import Lasso


# %% [Block 46] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)


# %% [Block 47] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
drop = torch.nn.Dropout(p=0.5)
model.eval()      # 추론 시 드롭아웃 자동 비활성화


# %% [Block 48] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
torch.nn.init.xavier_normal_(layer.weight)
W = np.random.randn(n_out, n_in) * np.sqrt(2 / (n_in + n_out))


# %% [Block 49] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


# %% [Block 50] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


# %% [Block 51] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
d = np.dot(a, b)
d = a @ b
d = torch.dot(a, b)


# %% [Block 52] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
C = A @ B
C = np.matmul(A, B)
C = torch.bmm(A, B)     # 배치 행렬곱


# %% [Block 53] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
At = A.T
At = torch.transpose(A, 0, 1)


# %% [Block 54] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
eigvals, eigvecs = np.linalg.eig(A)
vals, vecs = torch.linalg.eigh(A)   # 대칭행렬


# %% [Block 55] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
d = np.linalg.det(A - lam * np.eye(len(A)))
p = np.poly(A)      # 특성다항식 계수


# %% [Block 56] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
U, S, Vt = np.linalg.svd(A, full_matrices=False)
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]


# %% [Block 57] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
n = np.linalg.norm(A, ord='fro')
n = torch.linalg.matrix_norm(A, ord='fro')


# %% [Block 58] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
t = np.trace(A)
t = torch.trace(A)


# %% [Block 59] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
theta = np.linalg.inv(X.T @ X) @ X.T @ y
theta, *_ = np.linalg.lstsq(X, y, rcond=None)   # 더 안정적


# %% [Block 60] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
r = np.linalg.matrix_rank(A)
r = torch.linalg.matrix_rank(A)


# %% [Block 61] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
H = -np.sum(p * np.log2(p + 1e-12))
from scipy.stats import entropy
H = entropy(p, base=2)


# %% [Block 62] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
ce = -np.sum(p * np.log(q + 1e-12))
loss = torch.nn.CrossEntropyLoss()(logits, labels)


# %% [Block 63] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
kl = ce - H
kl = torch.nn.functional.kl_div(q.log(), p, reduction='batchmean')


# %% [Block 64] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from scipy.spatial.distance import jensenshannon
js = jensenshannon(p, q) ** 2


# %% [Block 65] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import mutual_info_score
mi = mutual_info_score(x_labels, y_labels)


# %% [Block 66] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
d = np.linalg.norm(a - b)
d = torch.cdist(A, B, p=2)


# %% [Block 67] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
d = np.sum(np.abs(a - b))
d = torch.cdist(A, B, p=1)


# %% [Block 68] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
sim = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)


# %% [Block 69] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
n = np.linalg.norm(x, ord=3)
n_inf = np.linalg.norm(x, ord=np.inf)


# %% [Block 70] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
j = len(A & B) / len(A | B)
from sklearn.metrics import jaccard_score


# %% [Block 71] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
out = torch.softmax(scores, dim=-1) @ V
out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)


# %% [Block 72] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
Q, K, V = x @ Wq, x @ Wk, x @ Wv
qkv = torch.nn.Linear(d_model, 3 * d_model)(x).chunk(3, dim=-1)


# %% [Block 73] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
mha = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8)
out, weights = mha(query, key, value)


# %% [Block 74] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
pos = np.arange(L)[:, None]
div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
pe = np.zeros((L, d)); pe[:, 0::2] = np.sin(pos * div)


# %% [Block 75] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
ffn = torch.nn.Sequential(
    torch.nn.Linear(512, 2048), torch.nn.ReLU(),
    torch.nn.Linear(2048, 512))


# %% [Block 76] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
x = ln(x + attn(x))
x = ln(x + ffn(x))


# %% [Block 77] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))


# %% [Block 78] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
alpha = torch.softmax(e, dim=-1)
assert torch.allclose(alpha.sum(-1), torch.ones(1))


# %% [Block 79] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
ppl = np.exp(cross_entropy_loss)
ppl = torch.exp(torch.nn.functional.cross_entropy(logits, targets))


# %% [Block 80] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


# %% [Block 81] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
G = sum(gamma ** k * r for k, r in enumerate(rewards))


# %% [Block 82] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
V = np.zeros(n_states)
V[s] = np.mean([compute_return(ep) for ep in episodes_from(s)])


# %% [Block 83] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
Q = np.zeros((n_states, n_actions))
best_action = np.argmax(Q[s])


# %% [Block 84] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
V[s] = sum(pi[s, a] * sum(P[s, a, s2] * (R[s, a] + gamma * V[s2])
           for s2 in states) for a in actions)


# %% [Block 85] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
V[s] = max(sum(P[s, a, s2] * (R[s, a] + gamma * V[s2])
           for s2 in states) for a in actions)


# %% [Block 86] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])


# %% [Block 87] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
delta = r + gamma * V[s_next] * (1 - done) - V[s]


# %% [Block 88] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
if np.random.rand() < eps:
    a = np.random.randint(n_actions)
else:
    a = np.argmax(Q[s])


# %% [Block 89] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
loss = -(log_probs * returns).mean()
loss.backward()


# %% [Block 90] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
adv = returns - values
adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 정규화


# %% [Block 91] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)


# %% [Block 92] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import precision_score
p = precision_score(y_true, y_pred)


# %% [Block 93] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import recall_score
r = recall_score(y_true, y_pred)


# %% [Block 94] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='macro')


# %% [Block 95] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, y_score)


# %% [Block 96] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)


# %% [Block 97] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
recon = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
loss = recon + kld


# %% [Block 98] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
std = torch.exp(0.5 * logvar)
z = mu + std * torch.randn_like(std)


# %% [Block 99] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
d_loss = bce(D(real), ones) + bce(D(G(z).detach()), zeros)
g_loss = bce(D(G(z)), ones)


# %% [Block 100] 01-02-D 🔵 🚀 ML 공식 100가지 낭독 훈련 — 읽는 법·기호·뜻·용도·파이썬
noise = torch.randn_like(x0)
xt = torch.sqrt(ab[t]) * x0 + torch.sqrt(1 - ab[t]) * noise
