import whisper

def test_whisper():
    model = whisper.load_model("tiny")
    result = model.transcribe("path_to_an_audio_file.wav")
    print(result["text"])

test_whisper()
