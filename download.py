from pathlib import Path
from typing import Any, Sequence
import ffmpeg
from yt_dlp import YoutubeDL
from huggingface_hub import hf_hub_download


def download_video(urls: Sequence[str]) -> list[dict[str, Any]]:
    if isinstance(urls, str):
        urls = [urls]
    __video_ext = "mp4"
    with YoutubeDL(
        {
            # 최고 품질 영상 mp4 & 최고 음질 m4a 로 받으나, 영상의 경우 FHD 이하로 제한한다.
            "format": f"bestvideo[height<=1080][ext={__video_ext}]+bestaudio[ext=m4a]/best[ext={__video_ext}]/best",
            "merge_output_format": __video_ext,
            "outtmpl": {"default": "dist/%(title)s.%(ext)s"},  # 제목.확장자 형식으로 저장
            # "outtmpl": {"default": __video_full_path},
            "throttledratelimit": 102400,
            "fragment_retries": 1000,
            # "overwrites": True,
            "concurrent_fragment_downloads": 3,  # 동시에 N개의 영상 조각을 다운로드
            "retry_sleep_functions": {"fragment": lambda n: n + 1}, # 다운로드 실패시 1초씩 증가시키면서 재시도
            # "progress_hooks": [call_back],  # 다운로드 진행 상황을 알려주는 콜백 함수
        }
    ) as ydl:
        ydl.download(urls)
        return [ydl.extract_info(url) for url in urls] # type: ignore

def download_audio(urls: Sequence[str]) -> list[dict[str, Any]]:
    if isinstance(urls, str):
        urls = [urls]
    with YoutubeDL(
        {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": {"default": "dist/%(title)s.%(ext)s"},
            "throttledratelimit": 102400,
            "fragment_retries": 1000,
            "concurrent_fragment_downloads": 3,
            "retry_sleep_functions": {"fragment": lambda n: n + 1},
        }
    ) as ydl:
        ydl.download(urls)
        return [ydl.extract_info(url) for url in urls] # type: ignore


def extract_audio(video_path: Path):
    audio_path = video_path.with_suffix(".wav")
    (
        ffmpeg
        .input(str(video_path))
        .output(str(audio_path), format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run()
    )
    return audio_path

def download_ggml_model(model_id: str):
    hf_hub_download(repo_id=model_id, filename='ggml-model.bin', local_dir='./models')


if __name__ == "__main__":
    url = input("Enter the YouTube video URL: ")
    model_id = input("Enter the model id: ")
    if url:
        video_info = download_video([url])
    if model_id:
        download_ggml_model(model_id)
