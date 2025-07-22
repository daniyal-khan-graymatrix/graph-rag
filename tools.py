from langchain_core.tools import tool
from typing import List
from graph_retriever import PDFDataRetriever
retriever = PDFDataRetriever()

@tool
def combined_pdf_query_tool(query: str) -> List[str]:
    """
    Returns structured (Neo4j) and semantic (MongoDB) results with source labels.
    """
    structured = retriever.structured_retriever(query)
    semantic = retriever.vector_similarity_retriever(query)
    
    combined = [f"[Graph] {res}" for res in structured] + [f"[Semantic] {res}" for res in semantic]
    return combined
