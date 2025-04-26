import streamlit as st
from dataclasses import dataclass
import os
import pyttsx3
import speech_recognition as sr
import time
import threading
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt_tab')
from typing import Literal
import numpy as np

# Interview duration set to 30 minutes (1800 seconds)
INTERVIEW_DURATION = 1800

def speak(text):
    """Convert text to speech in a separate thread."""
    def tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=tts)
    thread.start()

class VideoTransformer(VideoTransformerBase):
    """Video transformer for cheating detection using OpenCV DNN without dlib."""
    def init(self):
        # Load OpenCV DNN face detection model
        self.net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
        self.cheating_detected = False

    def transform(self, frame):
        """Process each video frame to detect cheating."""
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]

        # Prepare image for DNN face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        num_faces = 0
        looking_away = False

        # Process each detected face
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                num_faces += 1
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Simple side face detection based on bounding box asymmetry
                face_width = endX - startX
                face_height = endY - startY
                center_x = (startX + endX) // 2
                left_half = center_x - startX
                right_half = endX - center_x
                asymmetry_ratio = abs(left_half - right_half) / max(left_half, right_half)

                # If the face is significantly asymmetrical, assume looking away
                if asymmetry_ratio > 0.3:  # Threshold can be adjusted
                    looking_away = True
                    cv2.putText(img, "Looking away!", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display cheating indicators
        if num_faces == 0:
            cv2.putText(img, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if num_faces > 1:
            cv2.putText(img, "Multiple faces detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if looking_away:
            cv2.putText(img, "Looking away detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Determine cheating status
        self.cheating_detected = num_faces == 0 or num_faces > 1 or looking_away
        if not self.cheating_detected:
            cv2.putText(img, "Monitoring...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.session_state["cheating_detected"] = self.cheating_detected
        return img

def main_app():
    """Main application logic for the AI Interview System."""
    st.title("AI Interview System")

    position = st.selectbox("Select the position:", ["Data Analyst", "Software Engineer", "Cyber Security", "Web Development"])
    resume = st.file_uploader("Upload your resume", type=["pdf", "txt"])
    auto_play = st.checkbox("Let AI interviewer speak!")
    voice_input = st.checkbox("Use voice input for answers")
    
    st.sidebar.title("📊 Interview Progress")
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    st.sidebar.progress(st.session_state.question_count / 12)
    st.sidebar.write(f"*Question:* {st.session_state.question_count + 1}/12")

    with st.expander("📌 Instructions"):
        st.markdown("""
        - You have 30 minutes.
        - The webcam will monitor for cheating.
        - Answer in voice or text.
        - AI will ask resume, DSA and coding questions.
        """)

    # Initialize session state variables
    if "cheating_detected" not in st.session_state:
        st.session_state["cheating_detected"] = False
    if "cheating_warnings" not in st.session_state:
        st.session_state["cheating_warnings"] = 0
    if "time_up" not in st.session_state:
        st.session_state["time_up"] = False
    if "interview_stopped" not in st.session_state:
        st.session_state["interview_stopped"] = False
    if "waiting_for_ready" not in st.session_state:
        st.session_state["waiting_for_ready"] = False

    # Display remaining time
    if "start_time" in st.session_state:
        elapsed_time = time.time() - st.session_state.start_time
        if elapsed_time > INTERVIEW_DURATION:
            st.error("Interview time is up! The session has ended.")
            st.session_state["interview_stopped"] = True
            return
        remaining_time = max(0, INTERVIEW_DURATION - elapsed_time)
        minutes, seconds = divmod(int(remaining_time), 60)
        st.sidebar.write(f"Time Remaining: {minutes:02d}:{seconds:02d}")

    # Webcam monitoring
    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Handle cheating warnings
    if st.session_state["cheating_detected"]:
        st.session_state.cheating_warnings += 1
        if st.session_state.cheating_warnings >= 3:
            st.session_state.interview_stopped = True
        else:
            st.warning(f"Cheating detected! Warning {st.session_state.cheating_warnings}/3")

    @dataclass
    class Message:
        origin: Literal["human", "ai"]
        message: str

    def process_resume(resume):
        """Process the uploaded resume into a searchable vector store."""
        text = ""
        if resume.type == "application/pdf":
            pdf_reader = PdfReader(resume)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        else:
            text = resume.read().decode("utf-8")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(texts, embeddings)

    def initialize_session():
        """Initialize the interview session state."""
        if 'docsearch' not in st.session_state:
            st.session_state.docsearch = process_resume(resume)
        if 'retriever' not in st.session_state:
            st.session_state.retriever = st.session_state.docsearch.as_retriever(search_type="similarity")
        if "resume_history" not in st.session_state:
            st.session_state.resume_history = [Message("ai", "Tell me about yourself")]
            st.session_state.waiting_for_ready = True
            st.session_state.question_time = time.time()
        if "resume_memory" not in st.session_state:
            st.session_state.resume_memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        if "resume_screen" not in st.session_state:
            groq_api_key = os.getenv("GROQ_API_KEY", "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")
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
            PROMPT = PromptTemplate(
                input_variables=["history", "input"],
                template="""
                I am an AI interviewer conducting a structured technical interview. I will ask short and precise questions based on the candidate’s resume, followed by DSA and coding questions.

### Interview Structure:
1. Resume-Based Questions: Ask 2-3 line questions focused on key skills, projects, and experience.
2. DSA Questions: Ask three concise DSA questions (easy, medium, hard).
3. Coding Questions: Ask three short coding problems relevant to the candidate's job role.

### Interview Flow:
- Start by asking the candidate to introduce themselves.
- Ask skill-based questions, keeping them direct and efficient.
- Transition smoothly into DSA and coding questions.
- Ensure questions are engaging, relevant, and not overly lengthy.
- Give only review and rating of answer. Don't give updated code.

### Output Format:
- Use 2-3 line questions.
- Keep follow-ups precise and goal-oriented.
- If the candidate struggles, provide short hints instead of direct answers.

Let's start the interview. Ask the candidate to introduce themselves.
               
Current Conversation:
{history}

Candidate: {input}
                """
            )
            st.session_state.resume_screen = ConversationChain(
                llm=llm,
                memory=st.session_state.resume_memory,
                prompt=PROMPT,
                verbose=True
            )
            st.session_state.start_time = time.time()
            st.session_state.question_count = 0
            st.session_state.resume_chain = ConversationChain(
                llm=llm, memory=st.session_state.resume_memory, prompt=PROMPT, verbose=False
            )

        for msg in st.session_state.resume_history:
            with st.chat_message(msg.origin):
                st.markdown(msg.message)

        if st.session_state.question_count < 12:
            st.session_state.waiting_for_ready = st.button("Ready for next question")
            if st.session_state.waiting_for_ready:
                last_input = st.session_state.resume_history[-1].message
                ai_response = st.session_state.resume_chain.run(input=last_input)
                st.session_state.resume_history.append(Message("ai", ai_response))
                st.session_state.question_count += 1
                if auto_play:
                    speak(ai_response)
                st.rerun()
        else:
            st.success("✅ Interview Complete.")
            full_chat = "\n".join([f"{m.origin.capitalize()}: {m.message}" for m in st.session_state.resume_history])
            summary = st.session_state.resume_chain.run(input=f"Summarize this interview:\n{full_chat}")
            st.subheader("📝 Interview Summary")
            st.write(summary)

    def transcribe_audio():
        """Transcribe audio input from the user."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit=10)
            st.info("🔍 Processing your speech...")
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Transcribed: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand.")
            return ""

    def query_with_retry(chain, user_input, retries=3, delay=5):
        """Handle API queries with retry logic for rate limits."""
        for _ in range(retries):
            try:
                return chain.run(input=user_input)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise e
        return "Request failed due to repeated rate limit errors."

    def answer_callback():
        """Process the user's answer and generate the next AI question."""
        if st.session_state.question_count >= 12:
            st.write("Thank you for completing the interview!")
            return
        human_answer = st.session_state.get("answer", "")
        st.session_state.resume_history.append(Message("human", human_answer))
        with st.spinner("AI thinking..."):
            ai_response = query_with_retry(st.session_state.resume_screen, human_answer)
        st.session_state.resume_history.append(Message("ai", ai_response))
        st.session_state.question_count += 1
        if auto_play:
            speak(ai_response)
        st.session_state.waiting_for_ready = True
        st.session_state.question_time = time.time()

    if position and resume:
        initialize_session()

        # Check if interview should stop
        if st.session_state.interview_stopped:
            if st.session_state.time_up:
                st.error("You did not respond in time. Interview stopped.")
            elif st.session_state.cheating_warnings >= 3:
                st.error("Cheating detected three times. Interview stopped.")
            return

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.resume_history:
                with st.chat_message(msg.origin):
                    st.write(msg.message)

        # Handle response time enforcement
        if st.session_state.waiting_for_ready:
            with st.container():
                st.write("You have 10 seconds to start responding.")
                if st.button("Ready to Answer"):
                    if time.time() - st.session_state.question_time <= 10:
                        st.session_state.waiting_for_ready = False
                    else:
                        st.session_state.time_up = True
                        st.session_state.interview_stopped = True
        else:
            with st.container():
                if voice_input:
                    if st.button("🎤 Answer with Voice"):
                        transcribed_text = transcribe_audio()
                        if transcribed_text:
                            st.session_state["answer"] = transcribed_text
                            answer_callback()
                            st.rerun()
                user_input = st.text_area("Your Answer:", key="user_input", help="Enter your response here...", height=200)
                if st.button("Submit Text Answer"):
                    if user_input.strip():
                        st.session_state["answer"] = user_input
                        answer_callback()
                        st.rerun()
                    else:
                        st.warning("Please enter your answer before submitting.")

if __name__ == "__main__":
    main_app()
