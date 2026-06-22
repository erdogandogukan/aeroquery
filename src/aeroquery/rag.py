import ollama
import chromadb
from aeroquery.reports import REPORTS

def embed(text: str) -> list[float]:
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]


client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="reports")


def index_reports():
    
    for i, report in enumerate(REPORTS):
        
        vector = embed(report)
        collection.add(
            ids=[str(i)],
            embeddings=[vector],
            documents=[report],
        )
    print(f"{len(REPORTS)} rapor indekslendi.")



def search_reports(query: str, n: int = 3) -> list[str]:
    query_vector = embed(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n,
    )
    return results["documents"][0]

def answer_question(query: str) -> str:
   
    reports = search_reports(query, n=3)
    context = "\n".join(reports)

    prompt = f"""Use ONLY the following drone observation reports to answer the question.
If the reports do not contain relevant information, say so. Do not make up information.

Reports:
{context}

Question: {query}"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response["message"]["content"]