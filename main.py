import os
import pandas as pd
import warnings
from ast import literal_eval
from dotenv import load_dotenv
from openai import OpenAI
from Chatbot.create_db import create_dataframe
from Chatbot.create_db import split_long_answers
from Chatbot.create_db import add_embeddings_to_dataframe
from Chatbot.create_db import insert_data
from Chatbot.create_db import get_existing_collections
from Chatbot.build_chatbot import chatbot_controller


EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
DB_PATH = "./naver_data.db"
COLLECTION_1 = "answer"
COLLECTION_2 = "question"
LLM_MODEL = "gpt-3.5-turbo"
SYSTEM = """너는 한국 사람들이 가장 많이 이용하는 포털사이트 네이버의 서비스 스마트스토어에서 근무하는 도우미야. 
    스마트스토어에 입점하고자 하는 사람들의 질문을 받거나, 스마트스토어 규정 관련해서 아주 잘 알고있는 도우미지. 
    너는 사용자의 질문에 정확하고 친절하게 답해야해.

    [good example]
    유저 : 미성년자도 판매 회원 등록이 가능한가요?
    챗봇 : 
        네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
    유저 : 저는 만 18세입니다.
    챗봇 :
        만 14세 이상 ~ 만 19세 미만인 판매회원은 아래의 서류를 가입 신청단계에서 제출해주셔야 심사가 가능합니다.
        추가 설명 ~~

    [imappropriate question example]
    유저 : 오늘 저녁에 여의도 가려는데 맛집 추천좀 해줄래?
    챗봇 : 저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.

    만약 질문이 모호하다면 한 번 더 질문할 수 있어
    [good example]
    유저 : 절차를 알려줘
    챗봇 : 무슨 절차를 안내해드릴까요?
    유저 : 회원가입 절차를 알려줘
    챗봇 : 
    네이버 스마트스토어 회원가입 절차는 다음과 같아요:
    1. 네이버 아이디로 로그인
    2. 스마트스토어 이용약관 동의
    3. 사업자 정보 입력
    4. 상품 및 쇼핑몰 정보 입력
    5. 결제정보 입력
    6. 가입 완료
    
    자세한 내용은 네이버 스마트스토어 홈페이지에서 확인하실 수 있어요. 추가로 궁금한 점이 있으면 언제든지 물어봐주세요.

    너는 반드시 관련된 문맥(context)를 기반으로 대답해야해. 질문의 문맥은 아래와 같아
    그리고 반드시 이전에 했던 대화를 기반으로 대답해야해
    [context] =
    """


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    naver_data = pd.read_pickle("final_result.pkl")

    # naver FAQ 데이터 처리
    df = create_dataframe(naver_data)
    new_df = split_long_answers(df, max_length=512)

    if not os.path.exists("embeddings.csv"):
        print("embeddings.csv 파일이 없으므로 임베딩을 진행합니다")
        df_with_embeddings = add_embeddings_to_dataframe(new_df)
        df_with_embeddings.to_csv("new_embeddings.csv")
        new_df = df_with_embeddings.copy()
        new_df.reset_index(drop=True, inplace=True)
        new_df["id"] = new_df.index
        new_df["id"] = new_df["id"].astype(str)

    else:
        print("embeddings.csv 파일이 있으므로 넘어갑니다")
        new_df = pd.read_csv("embeddings.csv")
        new_df.reset_index(drop=True, inplace=True)
        new_df["id"] = new_df.index
        new_df["id"] = new_df["id"].astype(str)
        new_df["question_vector"] = new_df["question_vector"].apply(literal_eval)
        new_df["answer_vector"] = new_df["answer_vector"].apply(literal_eval)

    if not os.path.isdir("naver_data.db"):
        print("naver_data.db에 데이터 삽입을 진행합니다")
        question_collection, answer_collection = insert_data(
            new_df, DB_PATH, COLLECTION_1, COLLECTION_2, EMBEDDING_MODEL
        )
    else:
        print("기존에 존재하는 db를 가져옵니다")
        question_collection, answer_collection = get_existing_collections(
            DB_PATH, COLLECTION_1, COLLECTION_2
        )

    history = []
    chatbot_controller(
        new_df,
        question_collection,
        answer_collection,
        client,
        history,
        LLM_MODEL,
        SYSTEM,
    )
