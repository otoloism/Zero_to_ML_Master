# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #26 · 2부
NumPy 60줄이 PyTorch 15줄로 — 신경망 수학 오해 바로잡기와 GPT·CNN·확산 모델 속 같은 수학 (2부)

원문(책): 01-13 「ML 수학 연결 총정리 — 모든 것을 이어서! (2부)」 https://wikidocs.net/439749
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 26_2_numpy_to_pytorch.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다. [추가]/[수정] 표시가 있는 부분만 실행을 위해 보탰습니다.
"""

#======================================================================
# [1] 01-13-J 10단계 — 같은 신경망을 PyTorch로
#     ① 순전파 model(X) → ② 손실 loss_fn → optimizer.zero_grad() → ③ loss.backward() → ④ optimizer.step() 을 300번 반복합니다.
#     ⚠️ 책의 PyTorch 코드는 출력이 없어, 재현용 시드 고정(앞)과 최종 손실·예측 확인 출력(뒤)을 추가했습니다(표시된 부분).
#======================================================================
print("\n" + "=" * 60)
print("[1] 01-13-J 10단계 — 같은 신경망을 PyTorch로")
print("=" * 60)
# [추가 코드 — 책 본문에는 없음] 실행할 때마다 같은 결과가 나오도록 시드 고정
import torch
torch.manual_seed(0)

import torch
import torch.nn as nn

model = nn.Sequential(            # 🔨 층을 쌓는다 = 합성함수
    nn.Linear(2, 8),               # 🧱 xW1 + b1 (우리가 짠 그 줄)
    nn.ReLU(),                     # 🔨 np.maximum(0, x)
    nn.Linear(8, 2)                # 🧱 h1W2 + b2
)
loss_fn = nn.CrossEntropyLoss()   # 🎲 softmax + 교차엔트로피가 한 몸
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
T = torch.tensor([0, 1, 1, 0])   # 원-핫 대신 정답 인덱스만

for epoch in range(300):
    y_hat = model(X)              # ① 순전파  ← forward_pass()
    loss = loss_fn(y_hat, T)     # ② 손실     ← compute_loss()
    optimizer.zero_grad()       # ⚠️ 이전 그래디언트 청소 (필수!)
    loss.backward()             # ③ 역전파   ← 우리가 짠 5줄이 이 한 줄
    optimizer.step()            # ④ 업데이트 ← W -= lr * gW

# [추가 코드 — 책 본문에는 없음] 학습 결과 확인 — NumPy 버전처럼 4문제를 모두 맞히는지
print(f"최종 손실: {loss.item():.4f}")
with torch.no_grad():
    P = torch.softmax(model(X), dim=1)   # 로짓 → 확률 (CrossEntropyLoss 안에 숨어 있던 softmax)
for x_i, p_i, t_i in zip(X, P, T):
    print(f"입력 {x_i.tolist()} → 예측 {p_i.argmax().item()} (확신 {p_i.max().item():.3f})  정답 {t_i.item()}")
