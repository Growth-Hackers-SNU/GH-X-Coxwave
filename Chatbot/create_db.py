import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


# naver FAQ 데이터 처리 함수
def create_dataframe(raw_data: dict) -> pd.DataFrame:
    data = [
        (
            id + 1,
            question.replace("\n", " "),
            answer.replace("\xa0", "")
            .replace("\n", " ")
            .replace(
                " 위 도움말이 도움이 되었나요?   별점1점  별점2점  별점3점  별점4점  별점5점    소중한 의견을 남겨주시면 보완하도록 노력하겠습니다.  보내기 ",
                "",
            )
            .replace(" 도움말 닫기", ""),
        )
        for id, (question, answer) in enumerate(raw_data.items())
    ]
    df = pd.DataFrame(data, columns=["id", "question", "answer"])
    return df


# answer 길이가 긴 데이터프레임 처리 함수
def split_long_answer(answer, max_length=512):
    new_answers = []
    if len(answer) > max_length:
        num_splits = len(answer) // max_length + 1
        for i in range(num_splits):
            start_idx = i * max_length
            end_idx = (i + 1) * max_length
            new_answer = answer[start_idx:end_idx]
            new_answers.append(new_answer)
    else:
        new_answers.append(answer)
    return new_answers


# 기존 행을 복사한 후 수정
def split_long_answers(df, max_length=512):
    new_rows = []
    total_rows = len(df)
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for index, row in df.iterrows():
            answer = row["answer"]
            new_answer_list = split_long_answer(answer, max_length)
            for new_answer in new_answer_list:
                new_row = row.copy()
                new_row["answer"] = new_answer
                new_rows.append(new_row)
            pbar.update(1)
    df = pd.DataFrame(new_rows)
    return df


# 임베딩 생성 함수
def get_embedding(text, EMBEDDING_MODEL):
    model_checkpoint = EMBEDDING_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    outputs = model(**inputs)
    return outputs.last_hidden_state[0].detach().numpy()[0].tolist()


# 데이터프레임에 임베딩 column 생성 함수
def add_embeddings_to_dataframe(df):
    tqdm.pandas(desc="Calculating question embeddings")
    df["question_vector"] = df["question"].progress_apply(get_embedding)

    tqdm.pandas(desc="Calculating answer embeddings")
    df["answer_vector"] = df["answer"].progress_apply(get_embedding)

    return df


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model):
        self.model = embedding_model

    def __call__(self, emb_list: Documents) -> Embeddings:
        texts = emb_list  # 텍스트 리스트
        embeddings = [get_embedding(text, self.model) for text in texts]
        return embeddings


# chromadb에 데이터 삽입 함수
def insert_data(df, DB_PATH, COLLECTION_1, COLLECTION_2, EMBEDDING_MODEL):
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    answer_collection = chroma_client.create_collection(
        name=COLLECTION_1,
        metadata={"hnsw:space": "cosine"},
        embedding_function=MyEmbeddingFunction(EMBEDDING_MODEL),
    )
    question_collection = chroma_client.create_collection(
        name=COLLECTION_2,
        metadata={"hnsw:space": "cosine"},
        embedding_function=MyEmbeddingFunction(EMBEDDING_MODEL),
    )

    answer_collection.add(
        ids=df.id.tolist(),
        embeddings=df.answer_vector.tolist(),
    )

    question_collection.add(
        ids=df.id.tolist(),
        embeddings=df.question_vector.tolist(),
    )

    return question_collection, answer_collection


def get_existing_collections(DB_PATH, COLLECTION_1, COLLECTION_2):
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    answer_collection = chroma_client.get_collection(name=COLLECTION_1)
    question_collection = chroma_client.get_collection(name=COLLECTION_2)

    return question_collection, answer_collection
