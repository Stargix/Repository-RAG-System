from google import genai
import os
from retrieve import retrieve_repository
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")

def generate_answer(question):
    results = retrieve_repository(question, n_results=5)
    context = "\n\n".join([result["content"] for result in results])

    prompt = f"Answer this question: {question} directly. If this context files are helpful you can use them to reinforce your answer, answer concisely:\n\n{context}\n\n"
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)
    
    return response.text

# Example usage
answer = generate_answer("How is the sponsor dialog implemented?")
print(answer)