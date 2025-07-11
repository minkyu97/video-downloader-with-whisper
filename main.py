import pathlib
from pytubefix import YouTube
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline # type: ignore

url = input("Enter the YouTube video URL: ")

video = YouTube(url)
dist_dir = pathlib.PosixPath().cwd().joinpath("dist")

video_stream = video.streams.filter(progressive=True).get_highest_resolution()
audio_stream = video.streams.filter(only_audio=True).first()

assert video_stream is not None
assert audio_stream is not None

video_stream.download(output_path="dist")
audio_stream.download(filename_prefix="audio_", output_path="dist")
audio_file_path = audio_stream.get_file_path(filename_prefix="audio_", output_path="dist", file_system="macOS")

model_id = "openai/whisper-large-v3-turbo"
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

result = pipe(audio_file_path, chunk_length_s=30, generate_kwargs={"language": "en"}, return_timestamps=True)

def time_to_srt(total_time: float) -> str:
    total_seconds = int(total_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((total_time - total_seconds) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

srt_file_path = dist_dir.joinpath(video.title + ".srt")
srt_file_path.touch()
with open(srt_file_path, 'w') as srt_file:
    for i, chunk in enumerate(result["chunks"], 1): # type: ignore
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        srt_file.write(f"{i}\n{time_to_srt(start)} --> {time_to_srt(end)}\n{text}\n\n")

pathlib.PosixPath(audio_stream.get_file_path(filename_prefix="audio_", output_path="dist", file_system="macOS")).unlink()
