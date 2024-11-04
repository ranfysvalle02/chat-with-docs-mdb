import time
from flask import Flask, request, render_template, jsonify, session
import requests
import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.errors import ServerSelectionTimeoutError
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import concurrent.futures

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))  # Use environment variable for secret key if available
CHUNK_SIZE=2000
CHUNKS_PER_QUERY = 5
DATABASE_NAME = 'mydatabase'

client = MongoClient('mongodb://localhost/?directConnection=true')
db = client[DATABASE_NAME]

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)


# Define function to summarize each chunk
def summarize(txt):
    url = 'http://localhost:11434/v1/completions'
    headers = {'Content-Type': 'application/json'}
    full_prompt = f"""
[INST]
<<SYS>>
You are a helpful AI assistant who summarizes context and generates excellent questions.
<</SYS>>

[context to summarize]
{str(txt)}
[/context to summarize]

Summarize the context. Provide the summary, and some helpful questions that can be answered from the context.

[RESPONSE FORMAT]
- RESPOND IN PLAINTEXT (EMOJIS ARE ALLOWED). IMPORTANT!
- USE MARKDOWN LIST FORMAT.
- MAX RESPONSE LENGTH IS 1000 WORDS.
[/RESPONSE FORMAT]
[/INST]
"""
    data = {'prompt': full_prompt, 'model': 'llama3.2:3b', 'max_tokens': 5000}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['text']
    except requests.RequestException as e:
        return f"Error: {e}"


def get_collection_names():
    """Retrieve the names of all collections in the database."""
    collections = list(db.list_collection_names())
    app.logger.debug(f"Collections: {collections}")
    return collections
    
# https://www.llama.com/docs/model-cards-and-prompt-formats/meta-code-llama/
def generate_response(prompt, conversation_history):
    """Generate an AI response based on the prompt, conversation history, and file content."""
    formatted_history = "\n".join(conversation_history)
    full_prompt = f"""
[INST]
<<SYS>>
You are a helpful AI assistant.
<</SYS>>

[chat history]
{formatted_history}
[/chat history]

{prompt}
[/INST]
"""
    print("PROMPT: "+full_prompt)
    url = 'http://localhost:11434/v1/completions'
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': full_prompt, 'model': 'llama3.2:3b', 'max_tokens': 5000}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['text']
    except requests.RequestException as e:
        return f"Error: {e}"

def allowed_file(filename):
    """Check if the uploaded file has an allowed file extension."""
    ALLOWED_EXTENSIONS = {'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/status')
def get_mongo_status():
    """Microservice health check"""
    try:
        client = MongoClient('mongodb://localhost/?directConnection=true', serverSelectionTimeoutMS = 5000)
        client.server_info()
        return jsonify({"database_status": "🟢"})
    except pymongo.errors.ServerSelectionTimeoutError:
        return jsonify({"database_status": "🔴"}), 503


@app.route('/')
def index():
    return render_template('index.html', collections=get_collection_names())

@app.route('/ingest', methods=['POST'])
def ingest():
    text = str(request.json.get('text'))    
    collection_name = str(request.json.get('collection_name'))
    source = str(request.json.get('source'))
    chunk_size = request.json.get('chunk_size')
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Maximum size of each chunk
        chunk_overlap=20,  # Number of overlapping characters between chunks
        length_function=len,  # Function to determine the length of each chunk
        separators=["\n\n", "\n", " ", ""]  # Characters to use for splitting
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Convert chunks into Document objects
    docs = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
    if collection_name not in db.list_collection_names():
        return jsonify({'error': 'Collection not exists'})

    collection = db[collection_name]
    MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=collection, index_name="vector_index")
    return jsonify({'text': text, 'num_documents': len(docs)})


@app.route('/create_collection', methods=['POST'])
def create_collection():
    new_collection_name = request.json.get('name')
    if new_collection_name in db.list_collection_names():
        return jsonify({'error': 'Collection already exists'})
    db.create_collection(new_collection_name)
    # Create your index model, then create the search index
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine"
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    result = db[new_collection_name].create_search_index(model=search_index_model)
    print("New search index named " + result + " is building.")
    # Wait for initial sync to complete
    print("Polling to check if the index is ready. This may take up to a minute.")
    predicate = None
    if predicate is None:
        predicate = lambda index: index.get("queryable") is True
    while True:
        indices = list(db[new_collection_name].list_search_indexes("vector_index"))
        if len(indices) and predicate(indices[0]):
            break
        time.sleep(5)
    print(result + " is ready for querying.")
    return jsonify({'status': 'success', 'collections': get_collection_names()})

@app.route('/list_collections', methods=['GET'])
def get_collections():
    collections = get_collection_names()
    return jsonify({'collections': collections})

@app.route('/delete_collection', methods=['POST'])
def delete_collection():
    collection_name = request.json.get('name')
    if collection_name not in db.list_collection_names():
        return jsonify({'error': 'Collection does not exist'})
    db.drop_collection(collection_name)
    return jsonify({'status': 'success', 'collections': get_collection_names()})


@app.route('/explore', methods=['GET'])
def explore():
    collection_name = request.args.get('collection')
    if collection_name not in db.list_collection_names():
        return jsonify({'error': 'Collection does not exist'})
    collection = db[collection_name]
    documents = list(collection.aggregate([
        {
            "$match":{}
        },
        {"$project": {"_id": 0, "embedding": 0}}
    ], allowDiskUse=True))
    documents_summary = list(collection.aggregate([
        {
            '$group': {
                '_id': '$source', 
                'texts': {
                    '$push': '$text'
                }
            }
        }, {
            '$project': {
                'texts': {
                    '$slice': [
                        '$texts', 5
                    ]
                }
            }
        }
    ], allowDiskUse=True))
    return jsonify({'documents': documents, 'summary': summarize(str(documents_summary))})

@app.route('/update_chunk', methods=['POST'])
def update_chunk():
    action = request.json.get('action')
    collection_name = request.json.get('collection')
    source = request.json.get('source')
    og_text = request.json.get('og_text')
    new_text = request.json.get('new_text')

    if collection_name not in db.list_collection_names():
        return jsonify({'error': 'Collection does not exist'})

    collection = db[collection_name]

    if action == 'save':
        # get new embeddings
        new_embedding = embeddings.embed_documents([new_text])
        
        collection.update_many({'source': source, 'text': og_text}, {'$set': {'text': new_text, 'embedding': new_embedding}})

    elif action == 'delete':
        collection.delete_many({'source': source, 'text': og_text})

    return jsonify({'og_text': og_text, 'new_text': new_text})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    selected_collection = data.get('collection', '')
    chunk_count = int(str(data.get('chunk_count', CHUNKS_PER_QUERY)))
    print("CHUNK_COUNT: "+str(chunk_count))
    if selected_collection not in db.list_collection_names():
        conversation_history = session.get('conversation_history', [])
        conversation_history.append(f"Human: {user_input}")
        #no vector store
        query = user_input
        ai_response = generate_response(f"""
    [user input]
    {user_input}
    [/user input]

    [RESPONSE FORMAT]
        - RESPOND IN PLAINTEXT (EMOJIS ARE ALLOWED). IMPORTANT!
        - RESPOND TO THE [user input].
    """, conversation_history)

        conversation_history.append(f"AI: {ai_response}")
        session['conversation_history'] = conversation_history
        return jsonify({'response': ai_response, 'full_history': conversation_history})
    else:
            
        conversation_history = session.get('conversation_history', [])
        conversation_history.append(f"Human: {user_input}")
        # initialize vector store
        vectorStore = MongoDBAtlasVectorSearch(
            db[str(selected_collection)], embeddings, index_name="vector_index"
        )
        query = user_input
        # perform a search between the embedding of the query and the embeddings of the documents
        print("\nQuery Response:")
        print("---------------")
        docs = vectorStore.similarity_search(query, k=chunk_count)
        print(str(docs))
        print("---------------\n")
        ai_response = generate_response(f"""                                    
    [context]
    {str(docs)}
    [/context]

    USE THE [context] TO RESPOND TO [user_input=`{user_input}`]

    [RESPONSE FORMAT]
        - RESPOND IN PLAINTEXT (EMOJIS ARE ALLOWED). IMPORTANT!
        - USE THE CONTEXT TO RESPOND TO THE [user input].
        - THINK CRITICALLY AND STEP BY STEP. ONLY USE THE CONTEXT TO RESPOND.
        - DO NOT INCLUDE YOUR THOUGHT PROCESS IN YOUR RESPONSE. MAKE SURE YOUR RESPONSE IS COHERENT.
        - THINK CRITICALLY AND STEP BY STEP. ONLY USE THE CONTEXT TO RESPOND.
    [/RESPONSE FORMAT]
    """, conversation_history)

        conversation_history.append(f"AI: {ai_response}")

        session['conversation_history'] = conversation_history

        return jsonify({'response': ai_response, 'full_history': conversation_history, 'chunks': [doc.page_content for doc in docs]})
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['conversation_history'] = []
    return jsonify({'status': 'success', 'message': 'Chat history cleared'})
@app.route('/clear_all', methods=['POST'])
def clear_all():
    session.clear()
    return jsonify({'status': 'success', 'message': 'All data cleared'})

@app.route('/show_session', methods=['GET'])
def show_session():
    return jsonify(dict(session))
if __name__ == "__main__":
    app.run(debug=True)
