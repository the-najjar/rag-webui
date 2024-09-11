import argparse
import json
import logging
import os
import re

import httpx
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama

from config import llm_studio_chat_completion_url, llm_studio_model_id, ollama_model_id
from get_embedding_function import get_embedding_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_llm_studio = False

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the following context, which may include various types of data such as text, CSV, PDF, and more:

{context}

---

Question: {question}

Provide a comprehensive answer based on all the information given in the context, including any relevant data from CSV files or other structured data sources. If the answer includes numerical data or statistics, be sure to mention it.

After your answer, only list sources you used to find exact answer only (not context):
SOURCES:
["source_id", "source_id", "source_id", ...]

for example:
SOURCES:
["file_path/file_name:Page 1", "file_path/file_name:Line 5", "file_path/file_name:Key value"]
...

Return the answer in markdown format.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    response, sources = query_rag(query_text)
    print("Response:", response)
    print("\nSources:", sources)

def query_rag(query_text: str):
    embedding_function = get_embedding_function(
        use_llm_studio=use_llm_studio,
        model=llm_studio_model_id if use_llm_studio else ollama_model_id
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.max_marginal_relevance_search(query_text, k=10, fetch_k=20)

    context_text = "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in results
    ])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    logger.info(f"Generated prompt:\n{prompt}")

    if use_llm_studio:
        response_text = call_llm_studio_api(prompt)
    else:
        model = Ollama(model=ollama_model_id)
        response_text = model.invoke(prompt)

    # Extract the sources from the response
    sources = extract_sources(response_text)

    # Remove the sources from the response text
    response_text = re.sub(r'\n(?:\*{0,3}(?:SOURCES?|Sources?)[\s:]*\*{0,3})[\s\S]*?(?:\n\n|$)', '', response_text, flags=re.IGNORECASE)

    return response_text.strip(), sources

def extract_sources(response_text: str) -> list:
    try:
        sources_match = re.search(r'(?:\*{0,3}(?:SOURCES?|Sources?)[\s:]*\*{0,3})[\s\S]*?(\[.*?\])(?:\n\n|$)', response_text, re.IGNORECASE)
        if sources_match:
            sources_text = sources_match.group(1)
            sources = json.loads(sources_text)
            return [format_source(source) for source in sources]
        return []
    except Exception as e:
        logger.error(f"Failed to extract sources from response: {str(e)}")
        logger.error(f"Response text:\n{response_text}")
        return []


def format_source(source: str) -> str:
    parts = source.split(':')

    if len(parts) < 2:
        return source

    file_path_and_name = parts[0].split('/')
    if len(file_path_and_name) < 2:
        return source

    file_path = file_path_and_name[0]
    file_name = file_path_and_name[1]
    extension = os.path.splitext(file_name)[1].lower()

    first_part = parts[1]
    chunk = None

    if len(parts) > 2:
        chunk = ':'.join(parts[2:])

    if extension == '.pdf':
        response = f"{file_path}/{file_name}:Page {first_part}"
    elif extension in ['.xlsx', '.xls']:
        response = f"{file_path}/{file_name}:Sheet {first_part}:Row {parts[2]}"
    elif extension == '.txt':
        response = f"{file_path}/{file_name}:Line {first_part}"
    elif extension == '.json':
        response = f"{file_path}/{file_name}:Key {first_part}"
    else:
        response = f"{file_path}/{file_name}:{first_part}"

    if chunk and extension not in ['.xlsx', '.xls'] :
        response += f":{chunk}"
    elif chunk and extension in ['.xlsx', '.xls'] and len(parts) > 3:
        response += f":{':'.join(parts[3:])}"

    return response


def call_llm_studio_api(prompt: str) -> str:
    params = {
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant that provides comprehensive answers based on all available information, including structured data from CSV files and other sources."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
        "model": llm_studio_model_id,
    }
    try:
        with httpx.Client(timeout=None) as client:
            response = client.post(llm_studio_chat_completion_url, json=params)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {str(e)}")
    except KeyError as e:
        logger.error(f"Failed to find expected key in JSON response: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

    return "Sorry, I couldn't generate a response at this time."


if __name__ == "__main__":
    main()
