import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import requests

app = Flask(__name__)
CORS(app)

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
memory = ConversationBufferMemory()

@app.route("/models", methods=['GET'])
def get_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {groq_api_key}"}

    response = requests.get(url, headers=headers)
    data = response.json()['data']
    
    models = []
    for model in data:
        if 'whisper' not in model['id'] and 'vision' not in model['id']:
            models.append(model['id'])

    if response.status_code == 200:
        return jsonify({"models": models })
    else:
        return jsonify({"error": "Something went wrong"})

@app.route('/chat', methods=['POST'])
def run():
    data = request.get_json()
    query = data.get("query","")
    model = data.get("model","")
    
    # Make API request
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    )
    
    response = conversation.invoke({'input':f"Answer the query in a properly formatted markdown: {query}"})
    
    
    return jsonify({"answer": response['response']})

if __name__=="__main__":
    app.run(debug=True)