from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import os
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_docs(directory):
    loader = TextLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form.get('query')

        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Extract text from PDF
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                docs = split_docs(pages)
                
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = Chroma.from_documents(docs, embeddings)
                llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")

                docs = vectorstore.similarity_search(user_query)
                response = chain.run(input_documents=docs, question=user_query)
                
                return render_template('index.html', pdf_text=response)
        
        if user_query:
            
            chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key=OPENAI_API_KEY)
            response = chat.invoke(
    [
            HumanMessage(
                content=f"{user_query}"
            )
    ]
)           
    
            return render_template('index.html', query_response=response.content)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
