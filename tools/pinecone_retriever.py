from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4
import os

# ── Setup ──────────────────────────────────────────────────────────────────────

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")
PINECONE_ENV       = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
EMBEDDING_MODEL    = "text-embedding-3-small"   # OpenAI embedding model
TOP_K              = 5                           # number of chunks to retrieve
CHUNK_SIZE         = 500                         # characters per chunk
CHUNK_OVERLAP      = 50                          # overlap between chunks


class PineconeRetriever:
    """
    Handles document ingestion and retrieval using Pinecone vector DB.

    Two main operations:
      1. ingest_documents() — splits docs into chunks, embeds, stores in Pinecone
      2. retrieve()         — embeds user query, finds top K similar chunks
    """

    def __init__(self, namespace: str = "default"):
        # namespace isolates data per user/org in Pinecone
        self.namespace = namespace

        # OpenAI embeddings — converts text to vectors
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # text splitter — breaks large docs into smaller chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ) 

        # Pinecone client + index
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # create index if it does not exist
        if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,       # dimension for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
            )

        self.index = pc.Index(PINECONE_INDEX)

    def embed_text(self, text: str) -> list[float]:
        """Convert text into a vector using OpenAI embeddings."""
        return self.embeddings.embed_query(text)

    def ingest_documents(self, documents: list[Document]) -> int:
        """
        Split documents into chunks, embed each chunk, store in Pinecone.
        Returns the number of chunks stored.

        Usage:
          docs = [Document(page_content="...", metadata={"source": "policy.pdf"})]
          retriever.ingest_documents(docs)
        """
        chunks = self.splitter.split_documents(documents)

        vectors = []
        for chunk in chunks:
            vector = self.embed_text(chunk.page_content)
            vectors.append({
                "id": str(uuid4()),
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                },
            })

        # upsert in batches of 100 (Pinecone limit)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(
                vectors=vectors[i:i + batch_size],
                namespace=self.namespace,
            )

        return len(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Convert query to vector, search Pinecone, return top K chunks.

        Returns list of:
          {"text": "...", "source": "...", "score": 0.92}
        """
        query_vector = self.embed_text(query)

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        return [
            {
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("source", "unknown"),
                "score": round(match.score, 4),
            }
            for match in results.matches
        ]

    def format_for_prompt(self, query: str, top_k: int = TOP_K) -> str:
        """
        Retrieve chunks and format as context string for LLM prompt.

        Example output:
          "Relevant documents:
           [1] (source: policy.pdf, score: 0.92)
           Refunds are allowed within 30 days...

           [2] (source: faq.pdf, score: 0.87)
           To initiate a refund, contact support..."
        """
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return "No relevant documents found."

        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] (source: {chunk['source']}, score: {chunk['score']})\n{chunk['text']}"
            )
        return "Relevant documents:\n\n" + "\n\n".join(lines)
