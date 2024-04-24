import pickle
import re

import pandas as pd
import yaml
from LLM import embed
from tqdm import tqdm


# 주어진 답변에서 불필요한 내용 및 부호를 전처리하는 함수
def clean_unnecessary(answer: str, remove: list) -> str:

    for i in remove:
        answer = answer.replace(i, " ")  # 별점 부분 제거

    answer = re.sub("\s{2,}", "\n", answer)  # 빈칸이 긴 경우 줄이기
    answer = answer.replace("\xa0", "\n")  # \xa0를 \n으로 통일
    answer = answer.replace("'", "")  # 볼드 표시 제거

    return answer


# 긴 문자열을 짧게 나누는 함수
def chunk_string(long_string: str, chunk_size=500) -> list:

    chunks = []
    current_chunk = ""
    last_line = ""
    previous_line = ""

    for line in long_string.split("\n"):
        # 주어진 chunk_size를 초과하지 않는 한 계속해서 현재 chunk에 새로운 줄 추가
        if len(current_chunk) + len(line) <= chunk_size:
            current_chunk += line + "\n"
            last_line = line + "\n"
        # 현재 청크에 줄을 추가하면 chunk_size를 초과하게 되면 현재 청크 완성
        else:
            # 이전 chunk의 마지막 줄도 overlap되게 앞에 포함
            chunks.append((previous_line + current_chunk).rstrip("\n"))
            previous_line = last_line
            current_chunk = line + "\n"

    # 마지막으로 남은 chunk 처리
    if current_chunk:
        chunks.append((previous_line + current_chunk).rstrip("\n"))

    return chunks


if __name__ == "__main__":

    with open("chatbot/constants.yaml", "r") as yaml_file:
        constants = yaml.safe_load(yaml_file)

    # FAQ 데이터 불러오기
    print("Reading FAQ Data...")
    with open(constants["FAQ_DATA_FILE_PATH"], "rb") as file:
        loaded_data = pickle.load(file)

    df = pd.DataFrame(loaded_data.items())
    df.columns = ["Question", "Answer"]

    print("Sanitizing...")
    df["Answer"] = df["Answer"].apply(
        clean_unnecessary, remove=constants["DELETE_PARTS"]
    )
    df["Answer"] = df["Answer"].apply(chunk_string, chunk_size=constants["CHUNK_SIZE"])
    df = df.explode("Answer")
    df = df.reset_index(drop=True)

    df["QA"] = df["Question"] + " " + df["Answer"]
    df = df.drop(["Question", "Answer"], axis=1)

    tqdm.pandas(desc="Embedding Question-Answer Pairs")
    df["QA Vector"] = df["QA"].progress_apply(
        embed, model=constants["EMBED_MODEL_NAME"]
    )

    df.to_pickle("chatbot/vectordb.df")
