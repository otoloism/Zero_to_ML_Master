# -*- coding: utf-8 -*-
"""
명확한 프롬프트 (훨씬 나은 결과)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-07장 대규모 언어 모델(LLM) 이해하기 — GPT의 시대는 무엇이 다른가/04-07-03 GPT-3 사용의 도전 과제 — 천재 직원의 함정.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-07-03-A 1) 환각(Hallucination)
from openai import OpenAI

client = OpenAI()

# 환각이 발생하기 쉬운 질문: 모델이 정확히 알 수 없는 세부 정보를 요구
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "2024년 노벨 물리학상을 받은 한국인은 누구인가?"}
    ],
    temperature=0.7  # temperature가 높을수록 그럴듯한 오답이 나올 확률도 증가
)

# 주의: 모델은 이 질문에 "그런 사람은 없다"고 말할 수도, 없는 이름을 지어낼 수도 있다.
# 코드만으로는 이를 100% 방지할 수 없으며, 근본적 해결책은 04-08장의 RAG이다.
print(response.choices[0].message.content)


# %% [Block 2] 명확한 프롬프트 (훨씬 나은 결과)
vague_prompt = "이 코드 고쳐줘"

precise_prompt = """다음 Python 함수에서 발생하는 IndexError를 수정해줘.
- 입력: 빈 리스트가 들어올 수 있음
- 출력: 빈 리스트인 경우 None을 반환
- 함수 시그니처는 그대로 유지할 것
"""

# 두 프롬프트를 각각 호출해 결과를 비교하는 함수
def compare_prompts(prompts: list[str]):
    results = []
    for p in prompts:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": p}],
        )
        results.append(r.choices[0].message.content)  # 결과 저장
    return results

# precise_prompt 쪽이 일관되게 요구사항(예외 처리, 반환값)을 반영할 확률이 높다
outputs = compare_prompts([vague_prompt, precise_prompt])


# %% [Block 3] 04-07-03-C 3) 컨텍스트 윈도우(Context Window) 한계
import tiktoken  # OpenAI 공식 토크나이저 라이브러리

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    주어진 텍스트가 몇 개의 토큰으로 분해되는지 계산합니다.

    비유:
    긴 문장을 우체국에서 소포 여러 개로 나누어 보내는 것처럼,
    모델도 문장을 여러 개의 '토큰 소포'로 쪼개어 처리합니다.
    """
    encoding = tiktoken.encoding_for_model(model)  # 모델별 토크나이저 로드
    tokens = encoding.encode(text)                 # 문자열 → 토큰 ID 리스트
    return len(tokens)

CONTEXT_LIMIT = 16_000  # gpt-3.5-turbo 컨텍스트 윈도우(예시)

def remaining_budget(conversation: str, max_response: int = 500) -> int:
    used = count_tokens(conversation)
    return CONTEXT_LIMIT - used - max_response  # 음수면 대화를 줄여야 함


# %% [Block 4] 04-07-03-D 4) 비용과 지연시간(Latency)
def estimate_cost(input_tokens: int, output_tokens: int,
                   price_in_per_1k: float, price_out_per_1k: float) -> float:
    """
    호출 1회의 예상 비용(USD)을 계산합니다.
    가격은 모델·시점에 따라 달라지므로 실제 사용 시 최신 가격표를 확인하세요.
    """
    cost_in = (input_tokens / 1000) * price_in_per_1k
    cost_out = (output_tokens / 1000) * price_out_per_1k
    return cost_in + cost_out

# 예시 호출 (가격은 예시 값이며 실제 값이 아님)
total = estimate_cost(input_tokens=1200, output_tokens=300,
                       price_in_per_1k=0.0005, price_out_per_1k=0.0015)
print(f"예상 비용: ${total:.5f}")
