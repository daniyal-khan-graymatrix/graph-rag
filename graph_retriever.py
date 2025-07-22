from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema.embeddings import Embeddings
import os
import time
from typing import List, Dict
import numpy as np
from pymongo import MongoClient
from openai import OpenAI
import os
from dotenv import load_dotenv
from dotenv import load_dotenv
load_dotenv()

class PDFDataRetriever:
    def __init__(self, mongo_uri: str = os.getenv("MONGO_URI"),
                 neo4j_uri: str = os.getenv("NAG_NEO4J_URI"),
                 neo4j_auth: tuple = (os.getenv("NAG_NEO4J_USERNAME"), os.getenv("NAG_NEO4J_PASSWORD"))):
        """Initialize the PDFDataRetriever class."""
        self.entity_chain = self._setup_entity_chain()
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client["pdf_embeddings"]
        self.mongo_collection = self.mongo_db["documents"]
        self.embeddings = OpenAIEmbeddings()

    @staticmethod
    def _setup_entity_chain():
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting entities from the text. Each entity must include its name and type. "
                    "The types of entities you should consider include: Section, Organization, Website, "
                    "Address, Information, Service, Certification, Geographical Region, Purpose, Document, "
                    "Procedure, Event, Action, Government, Data, Product, Fee, Condition, Financial Value, "
                    "Command, Ability, Liability, Message, Responsibility, Documents, Communication, Rule, "
                    "Privilege, Circumstance, Reason, Regulation, Law, Task, Instruction, Network, Identity Document, "
                    "Regulations, Obligation, Statement, Rights, Legal System, Entities, etc. "
                    "Output entities in JSON format as a list of name-type pairs."),
            ("human", "Extract entities from the following question: {question}"),
        ])
        return prompt | ChatOpenAI(temperature=0, model_name="gpt-4.1", max_retries=3).with_structured_output(Entities)

    def generate_full_text_query(self, input: str) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str) -> List[str]:
        result = []
        try:
            entities = self.entity_chain.invoke({"question": question})
            print(f"Entities: {entities}")
            for entity in entities.entities:
                response = self.driver.execute_query(
                    """
                    CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                    YIELD node, score
                    WITH node
                    OPTIONAL MATCH (node)-[r1:RELATES_TO]->(neighbor1)
                    OPTIONAL MATCH (neighbor2)-[r2:RELATES_TO]->(node)
                    WITH 
                      collect(node.id + ' - ' + type(r1) + ' -> ' + neighbor1.id) + 
                      collect(neighbor2.id + ' - ' + type(r2) + ' -> ' + node.id) AS outputs
                    UNWIND outputs AS output
                    RETURN output
                    LIMIT 50
                    """,
                    query=entity.name
                )
                result.extend([record["output"] for record in response.records])
                print(f"response: {response}")
        except Exception as e:
            print(f"Error in structured_retriever: {e}")
        finally:
            self.driver.close()
        return result


        
    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        try:
            response = openai_client.embeddings.create(
                model=embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return []
        
    def vector_similarity_retriever(self, query: str) -> List[Dict]:
        """
        Vector similarity search in a single MongoDB collection.

        Args:
            collection_name: Name of the MongoDB collection.
            query: The query string to embed and compare.

        Returns:
            List of documents sorted by cosine similarity to the query.
        """
        try:
            collection = self.mongo_collection
            documents = list(collection.find())
            query_embedding = np.array(self.get_embedding(query))
            doc_scores = []

            for doc in documents:
                doc_embedding = np.array(doc.get("embedding", []))
                if doc_embedding.shape != query_embedding.shape:
                    continue
                try:
                    similarity = np.dot(doc_embedding, query_embedding) / (
                        np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding)
                    )
                    doc.pop("embedding", None)
                    doc_scores.append((similarity, doc))
                except Exception as e:
                    print(f"Skipping doc due to error: {e}")

            sorted_docs = sorted(doc_scores, key=lambda x: x[0], reverse=True)
            return [doc for _, doc in sorted_docs]

        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

class Entity(BaseModel):
    name: str = Field(...)
    type: str = Field(...)

class Entities(BaseModel):
    entities: List[Entity] = Field(...)



load_dotenv()

client = MongoClient(os.getenv("MongoUrl"))
db = client["soc_incidents"]

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-small"





