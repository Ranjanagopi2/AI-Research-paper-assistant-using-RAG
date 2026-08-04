from functools import partial
import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=partial(TextLoader, encoding="utf-8")
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents
def split_documents(documents):
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        for i, chunk in enumerate(chunks[:5]): 
            print(f"\n--Chunk {i+1} --")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n...and {len(chunks) - 5} more chunks.")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=10)

    print("--- Creating vector store ---")

    batch_size = 20

    vectorstore = None

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i+batch_size]

        print(f"Embedding batch {i//batch_size + 1}")

        if vectorstore is None:

            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )

        else:

            vectorstore.add_documents(batch)

    print(f"--- Finished creating vector store: saved to {persist_directory} ---")
    return vectorstore

def main():
    print("Main function")
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()