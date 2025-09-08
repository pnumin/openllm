import os
import time
import whisper
import ollama

# --- 1단계: STT 모듈 기능 (음성 -> 텍스트) ---
def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Whisper 모델을 사용하여 주어진 경로의 음성 파일을 텍스트로 변환합니다.
    """
    print(f"[STT] '{file_path}' 파일의 음성 인식을 시작합니다... (모델: {model_size})")
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        return f"오류: 음성 파일을 찾을 수 없습니다 - {file_path}"

    try:
        # Whisper 모델 로드 (폐쇄망에서는 모델 파일이 캐시에 있어야 함)
        model = whisper.load_model(model_size)
        
        # 음성 파일 변환 실행
        result = model.transcribe(file_path, fp16=False) # GPU가 없을 경우 fp16=False 권장
        
        print("[STT] 음성 인식 완료.")
        return result["text"]

    except Exception as e:
        return f"오류: 음성 인식 중 문제가 발생했습니다 - {e}"

# --- 2단계: Gemma3 요약 모듈 기능 (텍스트 -> 요약) ---
def summarize_text_with_gemma(text_to_summarize: str, model_name: str = "gemma3:4b") -> str:
    """
    Ollama를 통해 실행 중인 Gemma 모델에게 텍스트 요약을 요청합니다.
    """
    print(f"[LLM] '{model_name}' 모델을 사용하여 회의록 요약을 시작합니다...")

    # Gemma3 모델에게 역할을 부여하고 명확한 지시를 내리는 프롬프트
    prompt = f"""
    당신은 회의록을 전문적으로 요약하는 AI 어시스턴트입니다.
    아래에 제공되는 회의록 텍스트를 분석하여, 다음 세 가지 항목으로 구분하여 정리해 주세요.
    결과는 반드시 한국어로 작성해 주세요.

    1. **핵심 안건 요약**: 회의에서 논의된 주요 주제들을 간결하게 요약합니다.
    2. **주요 결정 사항**: 회의를 통해 결정된 내용들을 명확하게 정리합니다.
    3. **Action Items (담당자 및 기한)**: 앞으로 수행해야 할 작업, 담당자, 기한을 목록 형태로 정리합니다.

    ---
    [회의록 원본 텍스트]
    {text_to_summarize}
    ---

    위 형식에 맞춰 회의록을 요약해 주세요:
    """

    try:
        start_time = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        end_time = time.time()
        print(f"[LLM] 요약 생성 완료! (소요 시간: {end_time - start_time:.2f}초)")
        
        return response['message']['content']

    except Exception as e:
        return f"오류: Ollama API 호출 중 문제가 발생했습니다 - {e}"

# --- 3단계: 파이프라인 실행 로직 ---
def main_pipeline():
    """
    STT와 Gemma3 모듈을 순차적으로 실행하는 메인 파이프라인 함수.
    """
    # --- 설정 ---
    INPUT_AUDIO_FILE = "meeting.mp3"
    OUTPUT_FOLDER = "output"
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    TRANSCRIBED_TEXT_PATH = os.path.join(OUTPUT_FOLDER, "transcribed_text.txt")
    FINAL_SUMMARY_PATH = os.path.join(OUTPUT_FOLDER, "final_summary.txt")

    print("="*50)
    print("회의록 자동 생성 파이프라인을 시작합니다.")
    print("="*50)

    # --- 파이프라인 1단계: 음성 -> 텍스트 변환 실행 ---
    transcribed_text = transcribe_audio(INPUT_AUDIO_FILE)
    
    # STT 과정에서 오류가 발생했는지 확인
    if transcribed_text.startswith("오류:"):
        print(f"[파이프라인 중단] {transcribed_text}")
        return # 파이프라인 중단

    # 변환된 텍스트가 비어있는지 확인
    if not transcribed_text.strip():
        print("[파이프라인 중단] 음성 인식 결과, 텍스트가 비어있습니다.")
        return # 파이프라인 중단

    # 중간 결과물(STT 텍스트) 저장
    with open(TRANSCRIBED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(transcribed_text)
    print(f"-> STT 결과가 '{TRANSCRIBED_TEXT_PATH}'에 저장되었습니다.")
    
    print("\n" + "-"*50 + "\n")

    # --- 파이프라인 2단계: 텍스트 -> 요약본 생성 실행 ---
    summary_result = summarize_text_with_gemma(transcribed_text)

    # 요약 과정에서 오류가 발생했는지 확인
    if summary_result.startswith("오류:"):
        print(f"[파이프라인 중단] {summary_result}")
        return # 파이프라인 중단

    # 최종 결과물(요약본) 저장
    with open(FINAL_SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(summary_result)
    print(f"-> 최종 요약본이 '{FINAL_SUMMARY_PATH}'에 저장되었습니다.")
    
    print("\n" + "="*50)
    print("모든 파이프라인 작업이 성공적으로 완료되었습니다.")
    print("="*50)
    
    # 최종 결과 화면에 출력
    print("\n--- 최종 회의록 요약 ---")
    print(summary_result)


# --- 스크립트 실행 지점 ---
if __name__ == "__main__":
    # Ollama 서비스가 실행 중인지 먼저 확인하는 것이 좋습니다.
    main_pipeline()