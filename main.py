import pathlib
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline # type: ignore
from download import download_audio, download_video

url = input("Enter the YouTube video URL: ")
lang = input("Enter target video language (default: en): ") or "en"

download_video(url)
info = download_audio(url)
audio_file_path = pathlib.PosixPath(info["requested_downloads"][0]["filepath"]) # type: ignore
srt_file_path = audio_file_path.with_suffix(".srt")

model_id = "distil-whisper/distil-large-v3.5"
torch_dtype = torch.float32
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(str(audio_file_path), chunk_length_s=30, generate_kwargs={"language": lang}, return_timestamps=True)

def time_to_srt(total_time: float) -> str:
    total_seconds = int(total_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((total_time - total_seconds) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

srt_file_path.touch()
with open(srt_file_path, 'w') as srt_file:
    for i, chunk in enumerate(result["chunks"], 1): # type: ignore
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        srt_file.write(f"{i}\n{time_to_srt(start)} --> {time_to_srt(end)}\n{text}\n\n")

audio_file_path.unlink()