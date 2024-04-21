from enum import Enum


# 쿼리 생성함수
def query_collection(collection, query, max_results):
    results = collection.query(
        query_texts=query, n_results=max_results, include=["distances"]
    )
    return results


# context 생성 함수
def build_context(df, question, question_collection, answer_collection):
    context = ""

    # 사용자가 입력한 질문과 유사한 질문벡터 2개
    result = query_collection(question_collection, [question], max_results=2)
    searched_ids = result["ids"][0]
    searched_answers = (
        df[df["id"].isin(searched_ids)]["question"]
        + df[df["id"].isin(searched_ids)]["answer"]
    ).tolist()

    # 사용자가 입력한 질문과 유사한 대답벡터 1개
    result2 = query_collection(answer_collection, [question], max_results=1)
    searched_ids2 = result2["ids"][0]
    searched_answers2 = (
        df[df["id"].isin(searched_ids2)]["question"]
        + df[df["id"].isin(searched_ids2)]["answer"]
    ).tolist()

    searched_answers += searched_answers2

    for doc in searched_answers:
        context += str(doc) + "\n\n"
    return context


# 답변해주는 함수
def generate_response(messages, client, LLM_MODEL):
    result = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.1, max_tokens=500
    )
    print(result.choices[0].message.content + "\n")
    return result.choices[0].message.content


class ChatbotAction(Enum):
    CONTINUE = 1
    EXIT = 2


def chatbot(
    df,
    question,
    question_collection,
    answer_collection,
    client,
    history,
    LLM_MODEL,
    SYSTEM,
):

    context = build_context(df, question, question_collection, answer_collection)

    # 시스템 메시지에 이전 대화 내용을 포함하여 생성
    system = SYSTEM + context

    message = [{"role": "system", "content": system}]

    if len(history) != 0:
        for m in history[0]:
            if m["role"] == "user":
                message.append(m)
            if m["role"] == "assistant":
                message.append(m)

    # 사용자 질문 메시지 추가
    user_message = {"role": "user", "content": question}
    message.append(user_message)

    answer = generate_response(message, client, LLM_MODEL)

    history.append(message)
    history.append({"role": "assistant", "content": answer})

    return answer


def chatbot_controller(
    df, question_collection, answer_collection, client, history, LLM_MODEL, SYSTEM
):
    print("====================================================")
    print("챗봇 시작. 무엇을 도와드릴까요? 종료를 원하시면 '종료'를 입력하세요.")

    while True:
        question = input("질문 입력: ")
        action = handle_input(question)

        if action == ChatbotAction.EXIT:
            print("====================================================")
            print("챗봇 종료.")
            break
        elif action == ChatbotAction.CONTINUE:
            chatbot(
                df,
                question,
                question_collection,
                answer_collection,
                client,
                history,
                LLM_MODEL,
                SYSTEM,
            )


def handle_input(question):
    if question == "종료":
        return ChatbotAction.EXIT
    else:
        return ChatbotAction.CONTINUE
