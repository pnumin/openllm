import os
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# .env 파일 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ChromaDB 클라이언트 설정 (로컬에 저장)
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "rag_pdf_documents" # 컬렉션 이름 변경 (PDF 전용)

# 컬렉션 생성 또는 불러오기
try:
    collection = client.get_or_create_collection(name=collection_name)
except Exception as e:
    print(f"Error creating/getting collection: {e}")
    # 필요한 경우 컬렉션을 삭제하고 다시 생성 (예: 문서 업데이트 시)
    # client.delete_collection(name=collection_name)
    # collection = client.get_or_create_collection(name=collection_name)


def load_pdf_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(filepath)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n" # 페이지 구분 및 내용 추가
                    else:
                        print(f"경고: {filename} 파일의 {page_num+1} 페이지에서 텍스트를 추출할 수 없습니다. (스캔된 이미지일 수 있습니다.)")
                if text.strip(): # 비어있지 않은 텍스트만 추가
                    documents.append({"text": text.strip(), "source": filename})
                else:
                    print(f"경고: {filename} 파일에서 유효한 텍스트를 추출하지 못했습니다.")
            except Exception as e:
                print(f"PDF 파일 '{filename}' 처리 중 오류 발생: {e}")
        else:
            print(f"경고: '{filename}'은(는) PDF 파일이 아닙니다. 건너뜀.")
    return documents

def chunk_text(text, chunk_size=700, overlap=100):
    # 간단한 텍스트 청크 분할 함수
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        if start < 0:
            start = 0
    return chunks

# ------------------- 문서 로드 및 청크 분할 -------------------
doc_folder = "knowledge_base" # 지식 베이스 PDF 폴더 경로
if not os.path.exists(doc_folder):
    os.makedirs(doc_folder)
    print(f"'{doc_folder}' 폴더가 생성되었습니다. 이 안에 PDF 문서를 넣어주세요.")
    exit()

raw_docs = load_pdf_documents_from_folder(doc_folder)
if not raw_docs:
    print(f"'{doc_folder}' 폴더에서 처리할 PDF 문서를 찾을 수 없습니다. 문서를 추가해주세요.")
    exit()

all_chunks = []
chunk_metadata = []
chunk_ids = []
doc_id_counter = 0

for doc in raw_docs:
    text_chunks = chunk_text(doc["text"])
    for i, chunk in enumerate(text_chunks):
        chunk_id = f"{doc['source'].replace('.pdf', '')}_{doc_id_counter}_{i}"
        all_chunks.append(chunk)
        chunk_metadata.append({"source": doc["source"], "chunk_idx": i})
        chunk_ids.append(chunk_id)
    doc_id_counter += 1

print(f"총 {len(all_chunks)}개의 텍스트 청크를 생성했습니다.")

# ------------------- 임베딩 생성 및 ChromaDB에 저장 -------------------
embedding_model = "models/text-embedding-004"

def get_gemini_embeddings(texts):
    if not texts: # 입력 텍스트가 비어있는 경우 처리
        print("경고: 임베딩을 요청할 텍스트가 비어있습니다.")
        return []
    
    try:
        response = genai.embed_content(
            model=embedding_model,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        # 수정된 부분: 'embeddings' 대신 'embedding' 키를 직접 확인
        if 'embedding' not in response: # 'embedding' 키가 응답에 있는지 확인
            print(f"Gemini 임베딩 응답에 'embedding' 키가 없습니다.")
            print(f"전체 응답 내용: {response}") # 전체 응답 내용을 출력하여 문제 진단
            if 'error' in response: # 에러 정보가 포함되어 있다면 출력
                print(f"응답에 포함된 에러: {response['error']}")
            return []
        
        # 만약 입력 텍스트가 하나라면 embedding은 단일 리스트로, 여러 개라면 리스트의 리스트로 올 수 있습니다.
        # ChromaDB는 리스트의 리스트를 기대하므로, 단일 텍스트의 경우에도 리스트로 감싸줍니다.
        if isinstance(response['embedding'][0], float): # 단일 임베딩인 경우
            return [response['embedding']]
        else: # 여러 임베딩인 경우
            return response['embedding'] # 이미 리스트의 리스트 형태
            
    except Exception as e:
        print(f"Gemini 임베딩 생성 중 예외 발생: {e}")
        return []

# 이미 데이터가 있는지 확인하고 없으면 추가
if collection.count() == 0:
    print("ChromaDB에 데이터를 추가합니다...")
    batch_size = 10 # 한 번에 처리할 청크 수를 줄여서 테스트 (API 호출 제한 고려)
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = chunk_ids[i:i+batch_size]
        batch_metadata = chunk_metadata[i:i+batch_size]

        print(f"Batch {i//batch_size} (총 {len(batch_chunks)}개 청크) 임베딩 생성 시도...")
        embeddings = get_gemini_embeddings(batch_chunks)
        
        if embeddings:
            try:
                collection.add(
                    embeddings=embeddings,
                    documents=batch_chunks,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                print(f"Batch {i//batch_size}: {len(batch_chunks)}개의 청크를 ChromaDB에 추가했습니다. (총 {collection.count()}개)")
            except Exception as e:
                print(f"ChromaDB 추가 오류: {e}")
        else:
            print(f"Batch {i//batch_size}의 임베딩 생성에 실패했습니다. (임베딩이 비어있음)")
    print("모든 문서 임베딩 및 ChromaDB 저장 완료.")
else:
    print(f"ChromaDB에 이미 {collection.count()}개의 PDF 문서가 있습니다. 새로 추가하지 않습니다.")
    print("만약 문서를 업데이트하고 싶다면 'client.delete_collection(name=collection_name)'을 사용하여 기존 컬렉션을 삭제 후 다시 실행해주세요.")