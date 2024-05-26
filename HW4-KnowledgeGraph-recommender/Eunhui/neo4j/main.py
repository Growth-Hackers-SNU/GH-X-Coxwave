import json
from tqdm import tqdm
import yaml
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.pydantic_v1 import ValidationError
from neo4j_graph import (
    KnowledgeGraph,
    graph,
    data,
    map_to_base_node,
    map_to_base_relationship,
)


def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None, allowed_rels: Optional[List[str]] = None
):
    # YAML 파일 로드
    with open("prompts.yaml", "r") as file:
        prompts = yaml.safe_load(file)

    # allowed_nodes 및 allowed_rels를 프롬프트에 추가
    system_message = prompts["knowledge_graph"]["system"]
    if allowed_nodes:
        system_message += "\n- **Allowed Node Labels:** " + ", ".join(allowed_nodes)
    if allowed_rels:
        system_message += "\n- **Allowed Relationship Types:** " + ", ".join(
            allowed_rels
        )

    # ChatPromptTemplate 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", prompts["knowledge_graph"]["human"]),
            ("human", prompts["knowledge_graph"]["tip"]),
        ]
    )

    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)


def extract_and_store_graph(
    document: Document,
    nodes: Optional[List[str]] = None,
    rels: Optional[List[str]] = None,
) -> None:
    try:
        # Extract graph data using OpenAI functions
        extract_chain = get_extraction_chain(nodes, rels)
        data = extract_chain.invoke(document.page_content)["function"]

        # Debugging: Print extracted data for inspection
        print("Extracted Data:")
        print(data)

        # Construct a graph document
        graph_document = GraphDocument(
            nodes=[map_to_base_node(node) for node in data.nodes],
            relationships=[map_to_base_relationship(rel) for rel in data.rels],
            source=document,
        )

        # Store information into a graph
        graph.add_graph_documents([graph_document])

    except Exception as e:
        print(f"Error occurred: {e}")
        # Handle the error here, such as logging or skipping this document


# Load OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

allowed_nodes = ["Person", "Job", "Location", "Event", "Hobby", "Food"]
allowed_rels = [
    "like",
    "hate",
    "see",
    "hoby",
    "participated_in",
    "knows",
    "work_as",
    "visit",
    "interested_in",
]

for i, d in tqdm(enumerate(data), total=len(data)):
    extract_and_store_graph(d, allowed_nodes, allowed_rels)
print("완료")
