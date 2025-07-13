import pathlib
import subprocess
from typing import Sequence
from download import download_video, extract_audio


def process(urls: Sequence[str], lang: str):

    def _process_internal(filename: str):
        video_file_path = pathlib.PosixPath(filename)  # type: ignore
        srt_file_path = video_file_path.with_suffix(".srt")
        audio_file_path = extract_audio(video_file_path)

        # Step 2: whisper.cpp 실행 (SRT 출력)
        model_path = "models/ggml-model.bin"  # 모델 위치
        whisper_cpp_exec = "./whisper-cli"  # whisper.cpp 실행파일

        subprocess.run([
            whisper_cpp_exec,
            "--model", model_path,
            "--language", lang,
            "--output-srt", "true",
            "--output-file", str(video_file_path.with_suffix('')),
            "--max-len", "42",
            "--temperature", "0.1",
            str(audio_file_path)
        ], check=True)

        print(f"SRT saved to: {srt_file_path}")

        audio_file_path.unlink()

    download_video(urls, _process_internal)



if __name__ == "__main__":
    urls = filter(
        lambda s: len(s) > 0,
        map(
            lambda s: s.strip(),
            input("Enter URL (if multiple, seperate with spaces): ").split(" "),
        ),
    )
    lang = input("Enter target language (default en): ") or "en"

    process(list(urls), lang)
