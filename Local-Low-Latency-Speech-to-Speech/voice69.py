import os
import torch
import argparse
import pyaudio
import wave
from zipfile import ZipFile
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import openai
from openai import OpenAI
import time

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

print("Is CUDA available:", torch.cuda.is_available())

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
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_pth):
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se

    speaker_wav = audio_file_pth

    # Process text and generate audio
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)

        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = f'{output_dir}/output.wav'
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)

        print("Audio generated successfully.")
        play_audio(save_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")


def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    """
    Function to send a query to OpenAI's GPT-3.5-Turbo model, stream the response, and print each full line in yellow color.
    Logs the conversation to a file.
    """
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    temperature=1
    
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True
    )

    full_response = ""
    line_buffer = ""

    with open(chat_log_filename, "a") as log_file:  # Open the log file in append mode
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content

            if delta_content is not None:
                line_buffer += delta_content

                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(NEON_GREEN + line + RESET_COLOR)
                        full_response += line + '\n'
                        log_file.write(f"{bot_name}: {line}\n")  # Log the line with the bot's name
                    line_buffer = lines[-1]

        if line_buffer:
            print(NEON_GREEN + line_buffer + RESET_COLOR)
            full_response += line_buffer
            log_file.write(f"{bot_name}: {line_buffer}\n")  # Log the remaining line

    return full_response

def chatbot_conversation(num_exchanges):
    """
    Function to make two chatbots converse with each other, maintaining conversation history.
    Logs the conversation to a file.
    """
    conversation_history = []
    chatbot1_message = "Hello, im Julie, whats up?"
    chatbot2_message = ""
    chatbot1_system = open_file("chatbot2.txt")
    chatbot2_system = open_file("chatbot1.txt")

    for _ in range(num_exchanges):
        print(PINK + "Johnny:" + RESET_COLOR)
        chatbot2_message = chatgpt_streamed(chatbot1_message, chatbot1_system, conversation_history, "Johnny")
        conversation_history.append({"role": "user", "content": chatbot1_message})
        conversation_history.append({"role": "assistant", "content": chatbot2_message})
        prompt1 = chatbot2_message
        style = "default"
        audio_file_pth = "YOUR .mp3 PATH"
        process_and_play(prompt1, style, audio_file_pth)

        print(CYAN + "Julie:" + RESET_COLOR)
        chatbot1_message = chatgpt_streamed(chatbot2_message, chatbot2_system, conversation_history, "Julie")
        conversation_history.append({"role": "user", "content": chatbot2_message})
        conversation_history.append({"role": "assistant", "content": chatbot1_message})
        prompt2 = chatbot1_message
        style = "default"
        audio_file_pth2 = "YOUR .mp3 PATH"
        process_and_play(prompt2, style, audio_file_pth2)

        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

chatbot_conversation(100)  # Example call to start a conversation for 100 exchanges
