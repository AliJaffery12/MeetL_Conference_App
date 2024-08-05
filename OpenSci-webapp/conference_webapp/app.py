from flask import Flask, request, render_template, redirect, url_for, jsonify
import qrcode
import os
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import openai
import PyPDF2
from dotenv import load_dotenv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_FOLDER'] = 'static/'

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

# Initialize OpenAI API key
#os.environ["OPENAI_API_KEY"] = "sk-proj-wdSvDsZ40RLwnKPw6xezT3BlbkFJlyHwCY0fzC8U4GY4Wd3M"

# Initialize global variables
vector_store = None
qa_chain = None

def process_file(file_path):
    global vector_store, qa_chain
    
    # Load the document
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Create embeddings and store in Chroma
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    
    # Create a retrieval-based QA chain using GPT-4
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0,api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4})
    )

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global vector_store
    if request.method == 'POST':
        title = request.form['title']
        logo = request.files['logo']
        file = request.files['file']
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if logo:
                logo_filename = secure_filename(logo.filename)
                logo.save(os.path.join('static/', logo_filename))
            else:
                logo_filename = None
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            qr.add_data(url_for('chatbot', title=title, _external=True))
            qr.make(fit=True)
            img = qr.make_image(fill='black', back_color='white')
            img.save(os.path.join('static/', 'qrcode.png'))
            
            # Process the uploaded file
            try:
                process_file(file_path)
                if vector_store is None:
                    return jsonify({'error': 'Failed to initialize vector store'}), 500
            except Exception as e:
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
            return render_template('success.html', title=title, logo=logo_filename, qrcode='qrcode.png')
    
    return render_template('upload.html')
  

@app.route('/chatbot/<title>')
def chatbot(title):
    return render_template('chatbot.html', title=title)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    if qa_chain:
        try:
            answer = qa_chain.run(question)
            return jsonify({'answer': answer})
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                return jsonify({'answer': "I'm sorry, but the question and context exceed the model's maximum token limit. Please try asking a more specific question."})
            else:
                return jsonify({'answer': f"An error occurred: {str(e)}"})
        except Exception as e:
            return jsonify({'answer': f"An unexpected error occurred: {str(e)}"})
    else:
        return jsonify({'answer': "I'm sorry, but the conference information hasn't been loaded yet."})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/db_info')
def db_info():
    global vector_store
    if vector_store is not None:
        collection = vector_store._collection
        return jsonify({
            'status': 'initialized',
            'document_count': collection.count(),
            'sample_documents': collection.peek()
        })
    else:
        return jsonify({
            'status': 'not initialized',
            'error': 'Vector store not initialized. Please upload a file first.',
            'upload_url': url_for('upload_file', _external=True)
        })

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure upload and static folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    app.run(debug=True)