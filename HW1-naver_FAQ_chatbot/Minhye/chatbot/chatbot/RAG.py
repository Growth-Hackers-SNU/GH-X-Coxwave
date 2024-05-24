from typing import Union

import pandas as pd
from LLM import embed, generate
from sklearn.metrics.pairwise import cosine_similarity


# 지금까지의 대화와 새로운 유저 질문을 바탕으로 (문맥 검색에 사용할) 독립적인 질문을 생성하는 함수
def generate_question(
    prompt: str, model: str, user_question: str, chat_history: list
) -> str:
    prompt = prompt.format(
        chat_history=" ".join(chat_history), user_question=user_question
    )
    new_question = generate(model=model, messages=[{"role": "user", "content": prompt}])

    return new_question


# 유저의 질문을 대답하기 위한 문맥을 찾는 함수
def find_context(
    vector_db: pd.DataFrame,
    model: str,
    user_question: str,
    num_context: int,
    min_sim_score: float,
) -> Union[str, bool]:

    user_vector = embed(user_question, model, wait_time=0)

    cosine_similarities = cosine_similarity(
        vector_db["QA Vector"].tolist(), [user_vector]
    )

    # 가장 높은 유사도가 최소 요구 점수 미만이라면, False 반환
    if max(cosine_similarities) < min_sim_score:
        return False

    top_indices = cosine_similarities.flatten().argsort()[-num_context:][::-1]

    # 상위 k개의 문맥을 하나의 str로 합치기
    context = ""
    for i in range(num_context):
        context += f"\n Context {i+1} \n"
        context += vector_db["QA"].iloc[top_indices[i]]

    return context
