import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# Check if CUDA is installed
if torch.cuda.is_available():
    print("CUDA installed succesfully\n") 
else:
    print("CUDA not properly installed. Stopping process...")
    quit()
    
# Print available TTS models
view_models = input("View models? [y/n]\n")
if view_models == "y":
    tts_manager = TTS().list_models()
    all_models = tts_manager.list_models()
    print("TTS models:\n", all_models, "\n", sep = "")
    
# Prompt model selection
model = input("Enter model:\n")
    # for example, tts_models/multilingual/multi-dataset/xtts_v2

# Example voice cloning with selected model
tts = TTS((model), progress_bar=True).to(device)
tts.tts_to_file("This is a voice cloning test", speaker_wav="jett-train-audio.wav",
                language="en", file_path="output.wav")