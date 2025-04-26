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
from langchain_community.embeddingsmu import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt_tab')
from typing import Literal
import numpy as np
import mediapipe as mp

INTERVIEW_DURATION = 1800

def speak(text):
    def tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=tts)
    thread.start()

class VideoTransformer(VideoTransformerBase):
    def _init_(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            min_detection_confidence=0.5
        )
        self.cheating_detected = False
        self.pose_history = []

    def estimate_head_pose(self, face_landmarks, image_shape):
        model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype="double")
        landmark_indices = [1, 152, 33, 263, 61, 291]
        image_points = np.array([[face_landmarks.landmark[idx].x * image_shape[1],
                                  face_landmarks.landmark[idx].y * image_shape[0]]
                                 for idx in landmark_indices], dtype="double")
        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            if sy >= 1e-6:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                print(f"Head Pose - Yaw: {np.degrees(yaw):.2f}, Pitch: {np.degrees(pitch):.2f}, Roll: {np.degrees(roll):.2f}")
                return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
        print("Head pose estimation failed")
        return 0, 0, 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        num_faces = 0
        looking_away = False
        excessive_movement = False

        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            print(f"Number of faces detected: {num_faces}")
            for face_landmarks in results.multi_face_landmarks:
                yaw, pitch, roll = self.estimate_head_pose(face_landmarks, img.shape)
                if abs(yaw) > 15 or abs(pitch) > 10 or abs(roll) > 15:
                    looking_away = True
                    print("Looking away detected")
                self.pose_history.append((yaw, pitch, roll))
                if len(self.pose_history) > 5:
                    self.pose_history.pop(0)
                if len(self.pose_history) >= 5:
                    variances = [np.var([p[i] for p in self.pose_history]) for i in range(3)]
                    print(f"Pose variances: {variances}")
                    if any(var > 25 for var in variances):
                        excessive_movement = True
                        print("Excessive movement detected")
        else:
            print("No faces detected")

        self.cheating_detected = num_faces == 0 or num_faces > 1 or looking_away or excessive_movement
        print(f"Cheating detected: {self.cheating_detected}")
        st.session_state["cheating_detected"] = self.cheating_detected

        if num_faces == 0:
            cv2.putText(img, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif num_faces > 1:
            cv2.putText(img, "Multiple faces detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif looking_away:
            cv2.putText(img, "Looking away detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif excessive_movement:
            cv2.putText(img, "Excessive movement detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Monitoring...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main_app():
    st.title("AI Interview System")
    position = st.selectbox("Select the position:", ["Data Analyst", "Software Engineer", "Cyber Security", "Web Development"])
    resume = st.file_uploader("Upload your resume", type=["pdf", "txt"])
    auto_play = st.checkbox("Let AI interviewer speak!")
    voice_input = st.checkbox("Use voice input for answers")

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

    if "start_time" in st.session_state:
        elapsed_time = time.time() - st.session_state.start_time
        if elapsed_time > INTERVIEW_DURATION:
            st.error("Interview time is up! The session has ended.")
            st.session_state["interview_stopped"] = True
            return
        remaining_time = max(0, INTERVIEW_DURATION - elapsed_time)
        minutes, seconds = divmod(int(remaining_time), 60)
        st.sidebar.write(f"Time Remaining: {minutes:02d}:{seconds:02d}")

    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.write(f"Cheating detected status: {st.session_state['cheating_detected']}")  # Debug line
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
            groq_api_key = os.getenv("GROQ_API_KEY", "your_default_api_key_here")
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

    def transcribe_audio():
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

        if st.session_state.interview_stopped:
            if st.session_state.time_up:
                st.error("You did not respond in time. Interview stopped.")
            elif st.session_state.cheating_warnings >= 3:
                st.error("Cheating detected three times. Interview stopped.")
            return

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.resume_history:
                with st.chat_message(msg.origin):
                    st.write(msg.message)

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
