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
    parser.add_argument("--model", default="small", help="Whisper model name")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="output directory")
    parser.add_argument("--output_srt", type=str2bool, default=False, help="output srt file")
    parser.add_argument("--srt_only", type=str2bool, default=False, help="generate srt only")
    parser.add_argument("--verbose", type=str2bool, default=False, help="verbose progress")
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
    
    def transcribe_with_word_fix(audio_path):
        current_args = args.copy()
        if current_args.get("language") == "auto":
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            current_args["language"] = max(probs, key=probs.get)
            print(f"Detected language for word-level: {current_args['language']}")

        return model.transcribe(audio_path, word_timestamps=True, **current_args)

    subtitles = get_subtitles(
        audios, output_srt or srt_only, output_dir, 
        transcribe_with_word_fix
    )

    if srt_only: return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")
        
        # STYLE: Pure Outline Sempurna (No Shadow, No Ghosting)
        # Outline=2 (Ketebalan pas)
        # Spacing=0 & Blur=0 (Kunci biar outline rata di semua sisi)
        style = (
            "Fontname=Montserrat Black,Fontsize=20,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "Outline=3,Shadow=0,BorderStyle=1,"
            "Alignment=10,MarginV=10,Spacing=0,Blur=0"
        )

        print(f"Adding subtitles to {filename(path)}...")
        
        input_stream = ffmpeg.input(path)
        video = input_stream.video.filter('subtitles', srt_path, force_style=style)
        audio = input_stream.audio

        # TETEP PAKE KUALITAS MENTOK KANAN (CRF 16)
        ffmpeg.output(
            video, audio, out_path,
            vcodec='libx264',
            crf=16,
            preset='slow',
            pix_fmt='yuv420p',
            tune='film',
            acodec='copy'
        ).run(quiet=True, overwrite_output=True)

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
