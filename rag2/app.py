# app.py

from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import threading

app = Flask(__name__)

# --- 설정 부분 ---
DOCUMENTS_PATH = "documents" # PDF, MD 등 문서가 있는 폴더
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" 
VECTOR_DB_PATH = "faiss_index_gemma"

# RAG 체인을 저장할 전역 변수
rag_chain = None
rag_initialized = False
rag_init_lock = threading.Lock()

# --- RAG 시스템 구축 함수 ---
def setup_rag_system_background():
    global rag_chain, rag_initialized
    with rag_init_lock:
        if rag_initialized:
            print("RAG 시스템이 이미 초기화되었습니다.")
            return

        print("\n--- RAG 시스템 초기화 시작 (시간이 다소 소요될 수 있습니다) ---")
        try:
            print("1. 문서 로드 및 전처리 시작...")
            documents = []

            # 'documents' 폴더가 없으면 생성
            if not os.path.exists(DOCUMENTS_PATH):
                print(f"경고: '{DOCUMENTS_PATH}' 폴더를 찾을 수 없습니다. 폴더를 생성합니다.")
                os.makedirs(DOCUMENTS_PATH)

            # DOCUMENTS_PATH 폴더 내의 모든 파일을 확인
            for file_name in os.listdir(DOCUMENTS_PATH):
                file_path = os.path.join(DOCUMENTS_PATH, file_name)
                
                if file_path.endswith(".pdf"):
                    print(f"  - PDF 파일 로드 중: {file_path}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith(".md"):
                    print(f"  - Markdown 파일 로드 중: {file_path}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                # 다른 문서 형식(docx, txt 등)을 추가하고 싶다면 여기에 추가

            if not documents:
                print(f"경고: '{DOCUMENTS_PATH}' 폴더에 로드할 수 있는 문서가 없습니다. PDF 또는 MD 파일을 폴더에 추가해주세요.")
                return

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            split_documents = text_splitter.split_documents(documents)
            print(f"  - 원본 문서 {len(documents)}개, 분할된 문서 조각 {len(split_documents)}개.")

            print(f"2. 임베딩 모델 로드 및 벡터 저장소 생성/로드 시작...")
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

            if os.path.exists(VECTOR_DB_PATH):
                vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                print(f"  - 기존 벡터 저장소 '{VECTOR_DB_PATH}' 로드 완료.")
            else:
                vectorstore = FAISS.from_documents(split_documents, embeddings)
                vectorstore.save_local(VECTOR_DB_PATH)
                print(f"  - 새로운 벡터 저장소 '{VECTOR_DB_PATH}' 생성 및 저장 완료.")
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            print("  - 검색기(Retriever) 설정 완료.")

            llm = Ollama(model=OLLAMA_MODEL)
            print(f"3. LLM 모델 '{OLLAMA_MODEL}' 로드 완료.")

            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                "당신은 사내 내규에 대한 질문에 답변하는 친절하고 정확한 챗봇입니다. "
                "주어진 'context' 정보만을 사용하여 답변해주세요. "
                "만약 주어진 context 내에 답변할 수 있는 정보가 없다면, '주어진 정보로는 답변할 수 없습니다.'라고 솔직하게 말하세요. "
                "불확실한 정보나 추측을 포함하지 마세요.\n\n"
                "Context: {context}"),
                ("user", "{input}")
            ])

            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            print("4. RAG 체인 구축 완료. 챗봇을 사용할 준비가 되었습니다.")
            rag_initialized = True

        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류 발생: {e}")
            rag_chain = None
        print("--- RAG 시스템 초기화 종료 ---")

# Flask 앱 시작 시 RAG 시스템 초기화를 백그라운드 스레드에서 시작
# @app.before_first_request 는 Flask 2.3 버전부터 제거되었으므로,
# 앱 실행 시점에 스레드를 직접 시작하도록 변경합니다.
# Flask 앱이 완전히 시작될 때까지 기다릴 필요 없이,
# 스레드를 먼저 시작하고 Flask 앱을 실행합니다.
# Flask 앱 실행 시점에 이미 스레드가 동작 중이므로 첫 요청 처리 전에 RAG가 초기화될 것입니다.
thread = threading.Thread(target=setup_rag_system_background)
thread.start()

@app.route('/')
def index():
    return render_template('index.html', rag_status="초기화 중...")

@app.route('/chat', methods=['POST'])
def chat():
    global rag_chain, rag_initialized
    user_input = request.form['user_input']
    response_text = "챗봇 시스템이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요."
    
    if not rag_initialized:
        # RAG 초기화 상태를 다시 확인하여 클라이언트에게 전달
        return jsonify({'answer': "시스템 초기화 중입니다. 잠시만 기다려 주세요.", 'status': 'initializing'})

    if rag_chain:
        try:
            print(f"\n사용자 질문: {user_input}")
            response = rag_chain.invoke({"input": user_input})
            response_text = response['answer']
            print(f"챗봇 답변: {response_text}")
        except Exception as e:
            response_text = f"오류 발생: {e}. OLLAMA 서비스가 실행 중인지 확인해주세요."
            print(f"오류: {e}")
    
    return jsonify({'answer': response_text, 'status': 'ready'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)