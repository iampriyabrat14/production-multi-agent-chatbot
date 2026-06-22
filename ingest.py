from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from tools.pinecone_retriever import PineconeRetriever
import sys
import os

def ingest_pdf(pdf_path: str, namespace: str = "default"):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Pages loaded: {len(documents)}")

    retriever = PineconeRetriever(namespace=namespace)
    count = retriever.ingest_documents(documents)
    print(f"Chunks uploaded to Pinecone: {count}")
    print("Done!")

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/company_policy.pdf"
    ingest_pdf(pdf)
