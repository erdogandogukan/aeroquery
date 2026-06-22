import ollama
from aeroquery.storage import count_detections
from collections import Counter
from aeroquery.detect import get_detector
from aeroquery.rag import search_reports

def detect_image(image_path: str) -> str:
    detector = get_detector()
    detections = detector.predict(image_path)

    class_names = [d.class_name for d in detections]
    counts = Counter(class_names)

    summary = ", ".join(f"{count} {name}" for name, count in counts.items())
    return f"Detected: {summary}"

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

    detect_tool = {
    "type": "function",
    "function": {
        "name": "detect_image",
        "description": "Analyzes an image file and returns what objects are detected in it. Use this when the user asks what is in an image or wants an image analyzed.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The file path of the image to analyze, e.g. 'models/photo.jpg'"
                }
            },
            "required": ["image_path"]
        }
    }
}
    
    search_tool = {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": "Searches past drone observation reports for relevant information. Use this for open-ended questions about events, conditions, or observations, e.g. 'were there any suspicious activities?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what to look for in the reports."
                    }
                },
                "required": ["query"]
            }
        }
    }
    


    
    system_prompt = "You are AeroQuery, a database assistant for drone detection data. Answer the user's question using ONLY the data returned by the tools. Do not use your general knowledge or make up numbers. If a tool returns a count, state that exact number clearly and concisely."

    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

   
    response = ollama.chat(model = MODEL, messages=messages, tools=[count_tool, detect_tool, search_tool])

    
    messages.append(response["message"])

    # 3) Araç çağrısını al ve çalıştır
    tool_call = response["message"]["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"]["arguments"]
    

    if tool_name == "count_detections":
        result = count_detections(tool_args["class_name"])
    elif tool_name == "detect_image":
        result = detect_image(tool_args["image_path"])
    elif tool_name == "search_reports":
        result = search_reports(tool_args["query"])
        result = "\n".join(result)     
    else:
        result = "Unknown tool requested."        
    

    
    messages.append({"role": "tool", "content": str(result), "tool_name": tool_name})

    
    final_response = ollama.chat(model = MODEL, messages=messages)
    
    return final_response["message"]["content"]

