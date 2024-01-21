import config

import os
import torch
import argparse
import pyaudio
import wave
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import openai
from openai import OpenAI
import time
import speech_recognition as sr
import whisper
import multiprocessing
from multiprocessing import Pool

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(config.BASE_SPEAKERS_PATH + '/config.json', device=device)
en_base_speaker_tts.load_ckpt(config.BASE_SPEAKERS_PATH + '/checkpoint.pth')
tone_color_converter = ToneColorConverter(config.CONVERTER_PATH + '/config.json', device=device)
tone_color_converter.load_ckpt(config.CONVERTER_PATH + '/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(config.BASE_SPEAKERS_PATH + '/en_default_se.pth').to(device)
en_source_style_se = torch.load(config.BASE_SPEAKERS_PATH + '/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_pth):
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se

    speaker_wav = audio_file_pth

    # Process text and generate audio
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)

        src_path = config.OUTPUTS_PATH + '/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = config.OUTPUTS_PATH + '/output.wav'
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)

        print("Audio generated successfully.")
        play_audio(src_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")

# Wrapper function for multiprocessing
def process_and_play_wrapper(args):
    return process_and_play(*args)

# Function to record audio from the microphone and save to a file
def record_audio(file_name):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5  # Adjust the recording duration as needed

    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording stopped.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Output file path
    output_file_path = os.path.join(output_dir, file_name)

    # Write the recorded data to a file
    wave_file = wave.open(output_file_path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

# Function to handle a conversation with a user
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot2.txt") # personality
    pool = Pool(multiprocessing.cpu_count())  # Create a pool with as many processes as there are CPU cores
    tasks = []

    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break

        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Mistral:" + RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        
        prompt2 = chatbot_response
        style = "default"
        audio_file_pth2 = "output.wav"
        tasks.append((prompt2, style, audio_file_pth2))

        if len(tasks) >= 5:  # Process tasks in batches
            pool.map(process_and_play_wrapper, tasks)
            tasks = []  # Reset the task list after processing

    pool.close()
    pool.join()

user_chatbot_conversation()  # Start the conversation
