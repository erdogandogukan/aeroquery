import ollama
from aeroquery.storage import count_detections

MODEL = "llama3.1:8b"

def ask_agent(question: str) -> str:

    count_tool = {
        "type": "function",
        "function": {
            "name": "count_detections",
            "description": "Counts how many objects of a given class were detected and stored in the database. Use this to answer questions about how many of a specific object exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "The object class to count, e.g. 'car', 'pedestrian', 'bus', 'truck'"
                    }
                },
                "required": ["class_name"]
            }
        }
    }

    
    system_prompt = "You are AeroQuery, a database assistant for drone detection data. Answer the user's question using ONLY the data returned by the tools. Do not use your general knowledge or make up numbers. If a tool returns a count, state that exact number clearly and concisely."

    # Tek bir mesaj listesi — adım adım büyüyecek
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # 1) İlk çağrı: LLM'e soru + araç ver
    response = ollama.chat(model = MODEL, messages=messages, tools=[count_tool])

    # 2) LLM'in cevabını (araç çağırma kararını) listeye ekle
    messages.append(response["message"])

    # 3) Araç çağrısını al ve çalıştır
    tool_call = response["message"]["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"]["arguments"]
    

    result = count_detections(tool_args["class_name"])
    

    # 4) Araç sonucunu listeye ekle
    messages.append({"role": "tool", "content": str(result), "tool_name": tool_name})

    # 5) İkinci çağrı: artık liste her şeyi içeriyor → nihai cevap
    final_response = ollama.chat(model = MODEL, messages=messages)
    
    return final_response["message"]["content"]
