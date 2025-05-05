import requests
import streamlit as st


st.set_page_config(page_title="Audio Transcription")

# Displays a h1 header
st.title("Audio Transcription")

# Displays a file uploader widget
audio = st.file_uploader("Choose an audio file to upload", type=["mp3"])

# Displays text inputs for target languge option
# target_language = st.text_input("Target language", "ja")

# Displays a button
if st.button("Transcribe"):
    if audio is not None:
        files=(
            ('file', audio.getvalue()),
            ('model', (None, 'whisperX')),
            ('task', (None, 'translate')),
        )
        # Uploads the file to the server
        response = requests.post(f"{st.secrets['host']}/process", files=files)
        # Displays the response
        st.write(response.json()) 
        try: 
            for i, [start, end, text, speaker] in enumerate(response.json()["data"]["result"]):
                st.write(f"{speaker}: {text}")
        except Exception as e:
            st.error("Something went wrong. Please try again.")

