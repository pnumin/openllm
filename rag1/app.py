import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from flask import Flask, render_template, request, jsonify

# Flask 애플리케이션 초기화
app = Flask(__name__)

# --- 1. Google Gemini API 및 ChromaDB 설정 (이전 단계에서 했던 것과 동일) ---
# .env 파일 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

genai.configure(api_key=GOOGLE_API_KEY)

# Gemini LLM 모델 설정 (텍스트 생성용)
generation_model = genai.GenerativeModel('gemini-1.5-flash-latest') # 또는 'gemini-1.5-pro-latest'

# ChromaDB 클라이언트 설정 (prepare_pdf_knowledge_base.py에서 생성된 DB 경로 지정)
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "rag_pdf_documents" # prepare_pdf_knowledge_base.py에서 사용한 컬렉션 이름과 동일해야 함

try:
    collection = client.get_collection(name=collection_name)
    print(f"ChromaDB 컬렉션 '{collection_name}' 로드 성공. 문서 수: {collection.count()}개")
except Exception as e:
    print(f"ChromaDB 컬렉션을 찾을 수 없거나 로드 중 오류 발생: {e}")
    print("3단계 'prepare_pdf_knowledge_base.py'를 먼저 실행하여 PDF 지식 베이스를 구축해주세요.")
    # 실제 서비스에서는 에러 페이지를 보여주거나 다른 방식으로 처리해야 합니다.
    # 여기서는 앱이 시작될 때 문제가 있음을 알리고 진행합니다.
    collection = None # 컬렉션 로드 실패 시 None으로 설정

# Google Gemini의 임베딩 모델 사용 (질문 임베딩용)
embedding_model = "models/text-embedding-004"

# --- 2. RAG 핵심 로직 함수들 (이전 스크립트에서 가져옴) ---

def get_gemini_query_embedding(text):
    try:
        response = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return response['embedding']
    except Exception as e:
        print(f"Gemini 쿼리 임베딩 생성 오류: {e}")
        return None

def retrieve_documents(query, k=5): # k 값을 조절하여 검색할 청크 개수 지정
    if collection is None:
        print("경고: ChromaDB 컬렉션이 로드되지 않아 문서 검색을 수행할 수 없습니다.")
        return []

    query_embedding = get_gemini_query_embedding(query)
    if query_embedding is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"ChromaDB 문서 검색 오류: {e}")
        return []

def generate_response_with_rag(user_query):
    # 1. 문서 검색 (Retrieval)
    retrieved_chunks = retrieve_documents(user_query, k=5) # Flask 앱에서도 k 값 조절 가능

    if not retrieved_chunks:
        print("💡 관련 PDF 문서를 찾을 수 없습니다. 일반적인 정보로 답변을 시도합니다.")
        prompt = f"사용자의 질문에 답변해주세요: {user_query}"
    else:
        # 검색된 문서를 프롬프트에 추가 (Augmented Generation)
        context = "\n".join(retrieved_chunks)
        print("\n--- 검색된 PDF 지식 베이스 ---")
        for chunk in retrieved_chunks:
            print(f"- {chunk[:100]}...") # 처음 100자만 출력
        print("-------------------------\n")

        prompt = f"""당신은 유능한 전문 챗봇입니다. 다음의 '참고 정보'를 **반드시 활용**하여 '사용자의 질문'에 대해 **정확하고 간결하게 답변**해주세요.
        **오직 제공된 '참고 정보' 내에서만 답변해야 하며, 정보에 없는 내용은 절대 지어내지 말고 "해당 정보는 제가 가지고 있지 않습니다." 라고 명확히 답하세요.**
        추측하거나 일반적인 지식을 끌어와 답변하지 마세요. 답변은 한국어로 해주세요.

        **참고 정보:**
        {context}

        **사용자의 질문:**
        {user_query}

        **답변:**
        """

    # 2. 답변 생성 (Generation)
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini 답변 생성 오류: {e}")
        return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."

# --- 3. Flask 웹 서버 라우트(경로) 설정 ---

@app.route('/')
def index():
    """챗봇 메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """사용자 질문을 받아 RAG 챗봇 답변을 생성하고 반환합니다."""
    user_message = request.json.get('message') # 웹페이지에서 JSON 형태로 메시지를 받음
    if not user_message:
        return jsonify({"response": "질문을 입력해주세요."}), 400

    bot_response = generate_response_with_rag(user_message)
    return jsonify({"response": bot_response})

# Flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # 개발 모드: debug=True, 모든 IP에서 접근 가능: host='0.0.0.0'