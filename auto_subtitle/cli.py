import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from .utils import filename, str2bool, write_srt

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str, help="paths to video files")
    parser.add_argument("--model", default="small", help="Whisper model")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="output dir")
    parser.add_argument("--output_srt", type=str2bool, default=False, help="output srt")
    parser.add_argument("--srt_only", type=str2bool, default=False, help="srt only")
    parser.add_argument("--verbose", type=str2bool, default=False, help="verbose")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--language", type=str, default="auto")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    
    os.makedirs(output_dir, exist_ok=True)
    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    
    # Tambahkan word_timestamps=True untuk memunculkan teks satu persatu
    subtitles = get_subtitles(
        audios, output_srt or srt_only, output_dir, 
        lambda audio_path: model.transcribe(audio_path, word_timestamps=True, **args)
    )

    if srt_only: return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")
        
        # STYLE MONTESSERAT BLACK + STROKE + SHADOW
        # PrimaryColour: Putih (&H00FFFFFF), Outline: Hitam (&H00000000)
        # Alignment: 2 (Center Bottom), Outline: 3 (Setara Stroke tebal)
        style = (
            "Fontname=Montserrat Black,Fontsize=35,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "Outline=3,BorderStyle=1,Shadow=7,Alignment=2,MarginV=25"
        )

        print(f"Adding subtitles to {filename(path)}...")
        video = ffmpeg.input(path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style=style), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

def get_audio(paths):
    temp_dir = tempfile.gettempdir()
    audio_paths = {}
    for path in paths:
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")
        ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(quiet=True, overwrite_output=True)
        audio_paths[path] = output_path
    return audio_paths

def get_subtitles(audio_paths, output_srt, output_dir, transcribe):
    subtitles_path = {}
    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        result = transcribe(audio_path)
        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
        subtitles_path[path] = srt_path
    return subtitles_path

if __name__ == '__main__':
    main()
