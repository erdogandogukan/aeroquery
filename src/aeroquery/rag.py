import ollama
import chromadb
from aeroquery.reports import REPORTS
from datetime import datetime
from collections import Counter

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
    print(f"{len(REPORTS)} reports indexed.")

def add_report(report_text: str) -> None:
    
    vector = embed(report_text)
    report_id = datetime.now().isoformat()

    collection.add(
        ids=[report_id],
        embeddings=[vector],
        documents=[report_text],
    )

def create_report_from_detections(detections) -> str:
    class_names = [d.class_name for d in detections]
    counts = Counter(class_names)

    summary = ",".join(f"{count} {name}" for name, count in counts.items())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"{timestamp}, drone observation: detected {summary}."
    


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