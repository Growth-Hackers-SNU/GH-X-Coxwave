import pandas as pd
import yaml
from exceptions import ExitReason
from LLM import generate
from RAG import find_context, generate_question


def run_chatbot():
    # 대화 시작
    intro = constants["INTRO_TEXT"]
    print("챗봇:", intro)
    # 대화 내역 기록 시작 (실제 유저 질문 및 챗봇 답변만 저장)
    history = ["assistant: " + intro]
    # GPT 답변 지시 prompt
    messages = [{"role": "system", "content": prompts["INITIAL_PROMPT"]}]

    while True:
        try:
            # 질문 입력
            user_question = input("유저: ")
        except KeyboardInterrupt:
            print("Keyboard interrupt detected.")
            return ExitReason.NORMAL_EXIT

        # 유저가 quit을 입력하면 중단
        if user_question.lower() == "quit":
            break

        # 지금까지 질문이 두 개 이상 있었다면, 대화 내역을 바탕으로 독립적인 질문을 먼저 생성한 뒤 진행
        if len(messages) > 2:
            user_question = generate_question(
                prompts["FOLLOW_UP_PROMPT"],
                constants["GEN_MODEL_NAME"],
                user_question,
                history,
            )

        # 답변에 활용할 문맥 검색
        context = find_context(
            vector_db,
            constants["EMBED_MODEL_NAME"],
            user_question,
            num_context=constants["NUM_CONTEXT"],
            min_sim_score=constants["MIN_SIM_SCORE"],
        )

        # 관련 없는 질문이면 GPT로 답변 생성하지 않고 바로 다음 질문을 받는다
        if not context:
            print("\n챗봇:", constants["UNRELATED_QUESTION_MESSAGE"])
            continue

        # GPT에게 질문 및 문맥 제공
        messages.append(
            {"role": "user", "content": f"Question: {user_question} {context}"}
        )
        history.append(f"user: {user_question} /n ")
        answer = generate(model=constants["GEN_MODEL_NAME"], messages=messages)

        # 답변출력
        print(f"\n챗봇: {answer}\n")
        messages.append({"role": "assistant", "content": answer})
        history.append(f"assistant: {answer} /n ")


if __name__ == "__main__":
    with open("chatbot/constants.yaml", "r") as yaml_file:
        constants = yaml.safe_load(yaml_file)
    with open("chatbot/prompts.yaml", "r") as yaml_file:
        prompts = yaml.safe_load(yaml_file)

    vector_db = pd.read_pickle("chatbot/vectordb.df")

    exit_reason = run_chatbot()
    print("Exiting program with reason:", exit_reason.name)
