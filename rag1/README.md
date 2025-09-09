# RAG(Retrieval-Augmented Generation) 챗봇
+ 사용자의 질문이 들어오면, 미리 준비된 문서(지식 베이스, 여기서는 PDF 문서)에서 관련 정보를 '검색(Retrieval)'하여 찾아낸 후, 이 정보를 바탕으로 LLM(거대 언어 모델, 여기서는 Google Gemini)이 '답변을 생성(Generation)'

---
## 1단계: 개발 환경 설정(아나콘다사용)
1. 아나콘다(Anaconda) 설치: 아나콘다 공식 웹사이트 (https://www.anaconda.com/download)에서 본인의 운영체제(Windows, macOS, Linux)에 맞는 버전을 다운로드하여 설치
2. 새로운 가상 환경 생성: Anaconda Prompt에서 가상환경 생성
```bash
conda create -n rag_chatbot_env python=3.10
```
3. 가상 환경 활성화:
```bash
conda activate rag_chatbot_env
```
4. 필요한 라이브러리 설치
```bash
pip install google-generativeai  # Google Gemini API 사용 [6]
pip install chromadb           # 벡터 데이터베이스 (로컬 저장용) [21]
pip install pypdf              # PDF 파일 처리 (PDF 문서에서 텍스트 추출)
pip install python-dotenv      # API 키 등 민감 정보를 안전하게 관리
pip install Flask              # 챗봇서비스를 위한 웹 개발 라이브러리
```

## 2단계: Google Gemini API
+ .env 파일
```
GOOGLE_API_KEY="여기에_발급받은_API_키_입력"
```

## 3단계: 지식 베이스(Knowledge Base) 준비 및 임베딩 (PDF 파일 활용)
1.  knowledge_base라는 폴더를 만들고 그 안에 챗봇이 답변할 내용이 담긴 PDF 파일들을 저장
2. PDF 문서 로드, 텍스트 추출 및 분할 (Text Chunking):prepare_pdf_knowledge_base.py 실행
  + chroma_db 폴더가 생성되고, 그 안에 임베딩된 벡터와 문서 정보가 저장됩니다.

## 4단계: 챗봇 서비스 구현 (RAG 로직)
```
my_rag/
├── .env                  # Google API 키 저장 파일 (이미 존재)
├── chroma_db/            # ChromaDB 데이터 저장 폴더 (prepare_pdf_knowledge_base.py 실행 후 생성됨)
├── knowledge_base/       # PDF 문서 저장 폴더 (이미 존재)
├── app.py                # Flask 웹 애플리케이션의 메인 코드
├── prepare_pdf_knowledge_base.py # (이전 3단계 스크립트)
├── templates/            # HTML 파일들이 들어갈 폴더
│   └── index.html        # 챗봇 UI를 위한 HTML 파일
└── static/               # CSS, JavaScript, 이미지 등 정적 파일들이 들어갈 폴더
    └── style.css         # 챗봇 UI 스타일을 위한 CSS 파일 (선택 사항)
    └── script.js         # 챗봇 동작을 위한 JavaScript 파일
```
