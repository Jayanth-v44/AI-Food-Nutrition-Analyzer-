import os
import openai # type: ignore
from dotenv import load_dotenv # type: ignore
from loguru import logger # type: ignore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext # type: ignore
from llama_index.llms import OpenAI as LlamaOpenAI # type: ignore



# Load environment variables
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")

# Logging setup
logger.add("logs/app.log", rotation="10 MB")

service_context = ServiceContext.from_defaults(llm=LlamaOpenAI(model="gpt-4"))

def build_food_index(data_path: str = "data") -> VectorStoreIndex:
    try:
        documents = SimpleDirectoryReader(data_path).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        logger.info("Food knowledge index built successfully.")
        return index
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        return None
    


def query_nutrition_knowledge(question: str) -> str:
    try:
        index = build_food_index()
        if not index:
            return "Unable to build index."

        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        logger.info(f"Query executed: {question}")
        return str(response)
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        return "Unable to answer the question at the moment."
    


def get_nutrition_info(food_item: str) -> str:
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a food nutrition expert."},
                {"role": "user", "content": f"Give detailed nutrition info for: {food_item}"}
            ]
        )
        nutrition_data = response.choices[0].message.content
        logger.info(f"Fetched nutrition data for {food_item}")
        return nutrition_data
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        return "Unable to fetch nutrition info."