import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


app = Flask(__name__)
CORS(app)

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
memory = ConversationBufferMemory()

@app.route('/chat', methods=['POST'])
def run():
    query = request.json.get('query', "")
    data = {
    "model": "llama3-70b-8192",
    "messages": [
        {"role": "user", "content": f"Answer the following query in markdown format': {query}"}
        ]
    }
    
    # Make API request
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    )
    
    response = conversation.invoke({'input':f"Answer the query in a properly formatted markdown: {query}"})
    
    
    return jsonify({"answer": response['response']})

if __name__=="__main__":
    app.run(debug=True)