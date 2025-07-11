from yt_dlp import YoutubeDL


def download_video(url):
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
        ydl.download([url])
        return ydl.extract_info(url)

def download_audio(url):
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
        ydl.download([url])
        return ydl.extract_info(url)


if __name__ == "__main__":
    url = input("Enter the YouTube video URL: ")
    video_info = download_video(url)
    audio_info = download_audio(url)
