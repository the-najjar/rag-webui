import argparse
import logging
import os
import shutil
from typing import List

import pandas as pd
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, JSONLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_embedding_function import get_embedding_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def index_documents():
    documents = load_documents()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logger.info("✨ Clearing Database")
        clear_database()

    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents() -> List[Document]:
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file)

            try:
                if extension.lower() == '.pdf':
                    loader = PyPDFDirectoryLoader(root)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = f"data/{file}:{doc.metadata.get('page')}"
                    logger.info(f"Loaded {len(docs)} documents from PDF: {file}")
                    documents.extend(docs)
                elif extension.lower() == '.csv':
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                    for i, doc in enumerate(docs):
                        doc.metadata['source'] = f"data/{file}:{doc.metadata.get('row')}"  # Row number
                    logger.info(f"Loaded {len(docs)} documents from CSV: {file}")
                    documents.extend(docs)
                elif extension.lower() in ['.xlsx', '.xls']:
                    excel_file = pd.ExcelFile(file_path)
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name)
                        for i, row in df.iterrows():
                            doc = Document(page_content=row.to_dict().__str__(),
                                           metadata={'source': f"data/{file}:{sheet_name}:{i + 1}"})
                            documents.append(doc)
                        logger.info(f"Loaded {len(df)} rows from Excel: {file}, from sheet: {sheet_name}")
                    logger.info(f"Loaded {len(excel_file.sheet_names)} sheets from Excel: {file}")
                elif extension.lower() == '.txt':
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        doc = Document(page_content=line.strip(), metadata={'source': f"data/{file}:{i + 1}"})
                        documents.append(doc)
                    logger.info(f"Loaded {len(lines)} lines from Text: {file}")
                elif extension.lower() == '.json':
                    loader = JSONLoader(file_path, jq_schema='.', text_content=False)
                    docs = loader.load()
                    for i, doc in enumerate(docs):
                        doc.metadata['source'] = f"data/{file}:{i + 1}"  # Object number
                    logger.info(f"Loaded {len(docs)} documents from JSON: {file}")
                    documents.extend(docs)
                else:
                    logger.warning(f"Unsupported file type: {file}")
            except Exception as e:
                logger.error(f"Error loading file {file}: {str(e)}")

    logger.info(f"Loaded a total of {len(documents)} documents")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    chunked_documents = []
    for doc in documents:
        source = doc.metadata.get('source', '')
        file_name = os.path.basename(source)
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == '.pdf':
            chunked_documents.extend(split_pdf(doc))
        elif file_extension in ['.csv', '.xlsx', '.xls']:
            chunked_documents.extend(split_tabular(doc))
        elif file_extension == '.txt':
            chunked_documents.extend(split_text(doc))
        elif file_extension == '.json':
            chunked_documents.extend(split_json(doc))
        else:
            chunked_documents.extend(split_default(doc))

    return chunked_documents


def split_pdf(doc: Document) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(doc.page_content)
    return [
        Document(
            page_content=chunk,
            metadata={
                **doc.metadata,
                'chunk': i,
                'source': f"{doc.metadata['source']}:chunk_{i + 1}" if len(chunks) > 1 else doc.metadata['source']
            }
        )
        for i, chunk in enumerate(chunks)
    ]


def split_tabular(doc: Document) -> List[Document]:
    # Assuming each row is a separate document
    rows = doc.page_content.split('\n')
    return [
        Document(
            page_content=row,
            metadata={
                **doc.metadata,
                'row': i,
                'source': f"{doc.metadata['source']}:row_{i + 1}"
            }
        )
        for i, row in enumerate(rows)
    ]


def split_text(doc: Document) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(doc.page_content)
    return [
        Document(
            page_content=chunk,
            metadata={
                **doc.metadata,
                'chunk': i,
                'source': f"{doc.metadata['source']}:chunk_{i + 1}" if len(chunks) > 1 else doc.metadata['source']
            }
        )
        for i, chunk in enumerate(chunks)
    ]


def split_json(doc: Document) -> List[Document]:
    # Assuming each top-level key in the JSON is a separate document
    import json
    data = json.loads(doc.page_content)
    return [
        Document(
            page_content=json.dumps({key: value}),
            metadata={
                **doc.metadata,
                'key': key,
                'source': f"{doc.metadata['source']}:key_{key}"
            }
        )
        for key, value in data.items()
    ]


def split_default(doc: Document) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = splitter.split_text(doc.page_content)
    return [
        Document(
            page_content=chunk,
            metadata={
                **doc.metadata,
                'chunk': i,
                'source': f"{doc.metadata['source']}:chunk_{i + 1}" if len(chunks) > 1 else doc.metadata['source']
            }
        )
        for i, chunk in enumerate(chunks)
    ]


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        logger.info(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logger.info("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_source_id = source

        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source_id = current_source_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists('data'):
        shutil.rmtree('data')
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def check_if_existing_documents():
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    return existing_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index_documents()
