import streamlit as st
import os
import json
import base64
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image

# 1. 환경 변수 로드
load_dotenv()

# 2. Azure OpenAI 클라이언트 설정
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"),
    api_version="2024-05-01-preview", # 또는 사용 가능한 버전
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT")
)

# DALL-E용 클라이언트 (별도 설정이 필요한 경우)
# 만약 기본 client와 같다면 위 client를 그대로 사용해도 됩니다.
# 여기서는 예시로 별도 변수로 둡니다.
dalle_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"), # DALL-E 키가 다르다면 수정
    api_version="2024-04-01-preview",
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT") # DALL-E 엔드포인트가 다르다면 수정
)

# 배포 모델명 (환경변수 또는 직접 입력)
MODEL_CHAT = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o-mini") # 채팅 모델
MODEL_DALLE = "dall-e-3"      # 이미지 생성 모델
MODEL_TTS = "tts"             # TTS 모델 (배포명 확인 필요)

# --- 기능 함수 정의 ---

def get_ai_response(messages):
    """채팅 응답 생성"""
    response = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def text_to_speech(text):
    """TTS: 텍스트를 음성으로 변환"""
    try:
        response = client.audio.speech.create(
            model=MODEL_TTS,
            voice='shimmer', # soothing voice
            input=text
        )
        # 스트림릿에서 바로 재생하기 위해 바이너리 데이터 반환
        return response.content
    except Exception as e:
        st.error(f"TTS 오류: {e}")
        return None

def analyze_image_with_vision(image_bytes, user_prompt):
    """Vision: 이미지 분석 (관상/손금)"""
    encoded_image = base64.b64encode(image_bytes).decode('ascii')
    
    response = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": "당신은 관상과 손금을 잘 보는 신비한 타로 마스터입니다. 이미지의 특징을 분석해 운세를 점쳐주세요."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

def generate_dalle_image(prompt):
    """DALL-E: 이미지 생성"""
    try:
        result = dalle_client.images.generate(
            model=MODEL_DALLE,
            prompt=prompt + ", mystical tarot card style, high quality, fantasy art",
            n=1,
            style="vivid",
            quality="standard",
        )
        # 이미지 URL 반환
        return json.loads(result.model_dump_json())['data'][0]['url']
    except Exception as e:
        st.error(f"이미지 생성 오류: {e}")
        return None

# --- Streamlit UI 구성 ---

st.set_page_config(page_title="루미나 (Lumina): 당신의 운명을 비추는 별🔮", page_icon="🔮", layout="wide")

# 사이드바 설정
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
    st.title("🔮 Lumina")
    st.write("당신의 운명을 비추는 거울")
    
    st.markdown("---")
    mode = st.radio("서비스 선택", ["💬 타로 상담 (채팅)", "✋ 관상/손금 보기", "🎨 행운의 부적 만들기"])
    
    st.markdown("---")
    tts_enabled = st.checkbox("🔊 음성 답변 켜기", value=True)
    
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메인 타이틀
st.title(f"{mode}")

# 1. 타로 상담 (채팅) 모드
if mode == "💬 타로 상담 (채팅)":
    # 초기 시스템 메시지 설정
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "당신은 '루미나'라는 이름의 신비한 타로 마스터입니다. 말투는 신비롭고 예의 바르며, 비유적인 표현을 자주 사용합니다. 사용자에게 위로와 조언을 건네세요."}
        ]

    # 기존 대화 출력
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("방황하는 자여 고민을 말해 보시오..."):
        # 사용자 메시지 표시 및 저장
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("별들의 목소리를 듣고 있습니다..."):
                response_text = get_ai_response(st.session_state.messages)
                st.markdown(response_text)

                # 음성 재생 (TTS)
                if tts_enabled:
                    audio_bytes = text_to_speech(response_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        # AI 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# 2. 관상/손금 보기 모드
elif mode == "✋ 관상/손금 보기":
    st.info("당신의 손바닥이나 얼굴이 나온 사진을 올려주세요. 루미나가 운세를 분석해 드립니다.")

    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # 이미지 화면에 표시
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 사진', use_column_width=True)

        if st.button("운세 분석 시작하기"):
            with st.spinner("루미나가 갸우뚱한 표정으로 하지만 진지하게 살펴보고 있습니다..."):
                # 파일 바이트 읽기
                # 스트림릿 파일 객체는 read() 후 포인터가 이동하므로 주의
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()

                analysis_result = analyze_image_with_vision(image_bytes, "이 사람의 관상이나 손금을 보고 운세, 성격, 미래에 대한 조언을 신비로운 말투로 해줘.")

                st.success("분석 완료!")
                st.markdown(f"### 📜 루미나의 분석 결과 \n\n{analysis_result}")

                if tts_enabled:
                    audio_bytes = text_to_speech(analysis_result)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

# 3. 행운의 부적 만들기 모드
elif mode == "🎨 행운의 부적 만들기: (VIP서비스)":
    st.info("원하는 소원을 말하면, 당신만을 위한 행운의 부적을 그려줍니다.")

    wish = st.text_input("당신의 소원은 무엇인가요? (예: 취업 성공, 연애운 상승)")

    if st.button("부적 생성하기") and wish:
        with st.spinner("거대 우주의 마력을 끌어 모아 부적을 만들고 있습니다..."):
            image_url = generate_dalle_image(f"A mystic talisman symbol for {wish}")

            if image_url:
                st.image(image_url, caption=f"'{wish}'을(를) 기원하는 부적입니다.")
                st.success("이 이미지를 저장하여 부적으로 사용하세요!")
            else:
                st.error("마력이 부족해서 부적 만들기에 실패했습니다. 토큰을 충전해주세요")
