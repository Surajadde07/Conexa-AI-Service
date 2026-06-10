import whisper
import sys
import warnings

# Hide FP16 warning
warnings.filterwarnings("ignore")

# Load model
model = whisper.load_model("base")

# Get audio path
audio_path = sys.argv[1]

# Transcribe audio
result = model.transcribe(audio_path)

# Return ONLY transcript text
print(result["text"])