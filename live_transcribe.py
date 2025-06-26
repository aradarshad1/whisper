import whisper
import sounddevice as sd
import numpy as np
import torch

model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

while True:
    audio_data = record_audio()
    audio_data = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio_data).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=torch.cuda.is_available())
    result = whisper.decode(model, mel, options)
    print("You said:", result.text)