import pathlib
import subprocess
from typing import Sequence
from download import download_video, extract_audio


def process(urls: Sequence[str], lang: str):

    def _process_internal(file_path: str):
        video_file_path = pathlib.PosixPath(file_path)  # type: ignore
        srt_file_path = video_file_path.with_suffix(".srt")
        audio_file_path = extract_audio(video_file_path)

        # Step 2: whisper.cpp 실행 (SRT 출력)
        model_path = "models/ggml-model.bin"  # 모델 위치
        whisper_cpp_exec = "./whisper-cli"  # whisper.cpp 실행파일

        subprocess.run([
            whisper_cpp_exec,
            "-m", model_path,
            "-l", lang,
            "--output-srt", "true",
            "--output-file", str(video_file_path.with_suffix('')),
            str(audio_file_path)
        ], check=True)

        print(f"SRT saved to: {srt_file_path}")

        audio_file_path.unlink()

    infos = download_video(urls)

    for info in infos:
        if "requested_downloads" in info:
            _process_internal(info["requested_downloads"][0]["filepath"])
        elif "entries" in info:
            for entry in info["entries"]:
                _process_internal(entry["requested_downloads"][0]["filepath"])
        else:
            raise Exception("The video did not downloaded correctly")



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
