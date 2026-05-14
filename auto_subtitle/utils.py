import os
from typing import Iterator, TextIO

def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def write_srt(transcript: Iterator[dict], file: TextIO):
    # Logika untuk menulis per kata (word-level)
    count = 1
    for segment in transcript:
        if "words" in segment:
            for word in segment["words"]:
                start = format_timestamp(word["start"], always_include_hours=True)
                end = format_timestamp(word["end"], always_include_hours=True)
                text = word["word"].strip().upper() # Pakai Upper biar lebih tegas
                file.write(f"{count}\n{start} --> {end}\n{text}\n\n")
                count += 1
        else:
            # Fallback jika word_timestamps gagal
            start = format_timestamp(segment['start'], always_include_hours=True)
            end = format_timestamp(segment['end'], always_include_hours=True)
            file.write(f"{count}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
            count += 1

def filename(path):
    return os.path.splitext(os.path.basename(path))[0]
