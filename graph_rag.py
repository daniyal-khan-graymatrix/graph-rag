import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient
import pymupdf4llm
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

class PDFKnowledgeGraphBuilder:
    def __init__(
        self,
        neo4j_uri: str = os.getenv("NAG_NEO4J_URI"),
        neo4j_auth: tuple = (os.getenv("NAG_NEO4J_USERNAME"), os.getenv("NAG_NEO4J_PASSWORD")),
        mongo_uri: str = os.getenv("MONGO_URI"),
        mongo_db: str = "pdf_embeddings",
        mongo_collection: str = "documents"
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)

        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_collection = self.mongo_client[mongo_db][mongo_collection]

    def extract_text(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = pymupdf4llm.LlamaMarkdownReader()
        docs = reader.load_data(file_path=pdf_path)

        text = ""
        page_numbers = []

        for i, doc in enumerate(docs[:10]):  # Limit to first 10 pages
            text += doc.text
            sentences = re.split(r'(?<=[.!?]) +', doc.text)
            page_numbers.extend([i + 1] * len(sentences))

        return text, page_numbers

    def chunk_text(self, text: str, page_numbers: list, chunk_size=2000):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        current_page = page_numbers[0] if page_numbers else 1

        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append({"page_content": current_chunk.strip(), "page": current_page})
                current_chunk = sentence
                current_page = page_numbers[min(i, len(page_numbers) - 1)]

        if current_chunk:
            chunks.append({"page_content": current_chunk.strip(), "page": current_page})

        return chunks

    def create_knowledge_graph(self, documents):
        llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        transformer = LLMGraphTransformer(llm=llm)
        return transformer.convert_to_graph_documents(documents)

    def save_to_neo4j(self, graph_documents, file_name):
        with self.driver.session() as session:
            for doc in graph_documents:
                nodes = doc.nodes
                relationships = doc.relationships

                for node in nodes:
                    session.run(
                        """
                        MERGE (n:Entity {id: $id})
                        ON CREATE SET n.type = $type, n.file_name = $file_name
                        """,
                        id=node.id,
                        type=node.type,
                        file_name=file_name
                    )

                for rel in relationships:
                    session.run(
                        """
                        MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})
                        MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                        """,
                        src=rel.source.id,
                        dst=rel.target.id,
                        type=rel.type
                    )

    def store_embeddings_in_mongodb(self, chunks, file_name):
        embeddings = OpenAIEmbeddings()

        for chunk in chunks:
            vector = embeddings.embed_query(chunk["page_content"])
            self.mongo_collection.insert_one({
                "text": chunk["page_content"],
                "embedding": vector,
                "metadata": {
                    "file_name": file_name,
                    "page": chunk["page"]
                }
            })

    def run_for_pdf(self, pdf_path):
        file_name = os.path.basename(pdf_path)
        print(f"\n📄 Processing: {file_name}")
        try:
            text, page_numbers = self.extract_text(pdf_path)

            # Step 1: Knowledge Graph
            doc_for_kg = [Document(page_content=text, metadata={"file_name": file_name})]
            graph_documents = self.create_knowledge_graph(doc_for_kg)
            self.save_to_neo4j(graph_documents, file_name=file_name)

            # Step 2: Chunk → Embed → Store
            chunks = self.chunk_text(text, page_numbers)
            self.store_embeddings_in_mongodb(chunks, file_name)

            print(f"✅ Done: {file_name}")
        except Exception as e:
            print(f"❌ Failed to process {file_name}: {str(e)}")

    def run_for_all_pdfs(self, folder_path="data"):
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print("⚠️ No PDF files found in the data folder.")
            return

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            self.run_for_pdf(pdf_path)

# 🔁 Run pipeline for all PDFs in 'data' folder
if __name__ == "__main__":
    pipeline = PDFKnowledgeGraphBuilder()
    pipeline.run_for_all_pdfs()
