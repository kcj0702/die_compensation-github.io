# -*- coding: utf-8 -*-
"""실행 녹화를 해설 영상으로 만든다.

녹화 원본은 손이 멈춰 있는 시간이 길어 그대로 보여 주면 발표에서 못 쓴다.
구간마다 다른 배속으로 잘라 붙이고, 그 구간에서 무엇을 하는지와 그때 어떤
코드가 도는지를 자막으로 얹는다. 결과는 세 가지다.

  <이름>_영상.mp4        자막 없는 편집본 — HTML 해설 페이지가 이걸 쓴다
  <이름>_자막포함.mp4     자막을 구운 편집본 — PPT 에 넣거나 그냥 재생
  <이름>_해설.html       자막·목차·짚어주기 표시가 붙은 인터랙티브 페이지

쓰는 법:

    python build.py --video "녹화.mp4" --cues cues_sheet.py --name 보정치생성 \\
                    --out "C:/Users/.../실행영상_해설"

대본(--cues)은 SEGMENTS 리스트 하나만 있으면 된다. 형식은 cues_sheet.py 참고.

녹화 원본과 결과 mp4 에는 실제 품번과 보정 시트가 그대로 찍혀 있다.
회사 자료이므로 저장소에 커밋하지 않는다 — 이 폴더에는 대본과 도구만 둔다.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys


def ffmpeg_path() -> str:
    """imageio-ffmpeg 가 들고 오는 바이너리를 쓴다. 별도 설치가 필요 없다."""
    try:
        import imageio_ffmpeg
    except ImportError:  # pragma: no cover - 안내용
        raise SystemExit("imageio-ffmpeg 가 필요하다: pip install imageio-ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


FF = None  # main() 에서 채운다


def run(args: list[str], cwd: str | None = None) -> None:
    result = subprocess.run(args, capture_output=True, cwd=cwd)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", "replace")[-3000:])
        raise SystemExit("ffmpeg 실패: %s" % args[-1])


def duration(path: str) -> float:
    """번들에 ffprobe 가 없어서 ffmpeg 가 찍는 Duration 줄을 읽는다."""
    result = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True)
    text = result.stderr.decode("utf-8", "replace")
    hit = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    if not hit:
        raise SystemExit("길이를 읽지 못했다: %s" % path)
    hours, minutes, seconds = hit.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_segments(path: str) -> list:
    spec = importlib.util.spec_from_file_location("walkthrough_cues", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SEGMENTS


def cut(source: str, segments: list, segdir: str, recut: bool) -> list[str]:
    """구간마다 배속을 다르게 줘서 따로 인코딩한다.

    한 번에 filter_complex 로 묶으면 split 버퍼가 커져 터진다. 구간을 파일로
    떨어뜨린 뒤 concat 디먹서로 붙이는 편이 안전하고, 다시 손볼 때 바뀐 구간만
    다시 자를 수 있다.
    """
    os.makedirs(segdir, exist_ok=True)
    parts = []
    for index, (start, end, speed, *_rest) in enumerate(segments):
        out = os.path.join(segdir, "s%02d.mp4" % index)
        if os.path.exists(out) and not recut:
            parts.append(out)
            continue
        run([FF, "-hide_banner", "-loglevel", "error",
             "-ss", "%.3f" % start, "-to", "%.3f" % end, "-i", source,
             "-an", "-vf", "setpts=PTS/%.4f,fps=30,format=yuv420p" % speed,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-y", out])
        parts.append(out)
        print("  %02d  %6.1f~%6.1f  %.2f배속  ->  %.2f초"
              % (index, start, end, speed, duration(out)))
    return parts


def concat(parts: list[str], segdir: str, out: str) -> None:
    """중간 파일이라 화질을 아끼지 않는다 — 마지막 인코딩은 아래 encode/burn 이다."""
    listing = os.path.join(segdir, "list.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for part in parts:
            handle.write("file '%s'\n" % part.replace(chr(92), "/"))
    run([FF, "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listing, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-y", out])


def timeline(segments: list, parts: list[str]) -> tuple[list[dict], float]:
    """자막 시각은 계산값이 아니라 실제로 만들어진 구간 길이로 잡는다.

    setpts 는 프레임 경계에서 반올림되기 때문에 (end-start)/speed 로 쌓으면
    뒤로 갈수록 자막이 밀린다.
    """
    cues, clock = [], 0.0
    for (start, end, speed, text, note, mark), path in zip(segments, parts):
        length = duration(path)
        cues.append({"start": round(clock, 3), "end": round(clock + length, 3),
                     "text": text, "note": note, "speed": speed,
                     "srcStart": start, "srcEnd": end,
                     "mark": ({"x": mark[0], "y": mark[1]} if mark else None)})
        clock += length
    return cues, clock


def stamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60,
                                    ms // 1000 % 60, ms % 1000)


def write_srt(cues: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for number, cue in enumerate(cues, 1):
            handle.write("%d\n%s --> %s\n%s\n%s\n\n"
                         % (number, stamp(cue["start"]), stamp(cue["end"]),
                            cue["text"], cue["note"]))


def write_vtt(cues: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")
        for cue in cues:
            handle.write("%s --> %s\n%s\n\n"
                         % (stamp(cue["start"]).replace(",", "."),
                            stamp(cue["end"]).replace(",", "."), cue["text"]))


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 956
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Malgun Gothic,38,&H00FFFFFF,&H00FFFFFF,&H00201814,&HB0120C08,0,0,0,0,100,100,0,0,3,10,0,2,120,120,34,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_stamp(seconds: float) -> str:
    cs = int(round(seconds * 100))
    return "%d:%02d:%02d.%02d" % (cs // 360000, cs // 6000 % 60,
                                  cs // 100 % 60, cs % 100)


def write_ass(cues: list[dict], path: str) -> None:
    """자막은 SRT 가 아니라 ASS 로 굽는다.

    큰 줄(무엇을 하는지)과 작은 줄(어떤 코드가 도는지)의 크기·색이 같으면 두
    줄이 한 덩어리로 뭉쳐 읽히지 않는다. SRT + force_style 로는 줄마다 다른
    스타일을 줄 수 없어서 인라인 태그를 쓴다.
    """
    note_style = "{" + chr(92) + "fs27" + chr(92) + "c&H00DED0BF&}"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(ASS_HEAD)
        for cue in cues:
            body = cue["text"] + chr(92) + "N" + note_style + cue["note"]
            handle.write("Dialogue: 0,%s,%s,Main,,0,0,0,,%s\n"
                         % (ass_stamp(cue["start"]), ass_stamp(cue["end"]), body))


def encode(source: str, out: str) -> None:
    run([FF, "-hide_banner", "-loglevel", "error", "-i", source,
         "-c:v", "libx264", "-preset", "slow", "-crf", "25",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", out])


def burn(source: str, ass: str, out: str) -> None:
    """자막 경로에 드라이브 문자(콜론)가 들어가면 필터 인자가 쪼개진다.

    작업 폴더를 자막 옆으로 옮기고 파일 이름만 넘겨 콜론을 피한다.
    """
    run([FF, "-hide_banner", "-loglevel", "error", "-i", source,
         "-vf", "subtitles=%s" % os.path.basename(ass),
         "-c:v", "libx264", "-preset", "slow", "-crf", "25",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", out],
        cwd=os.path.dirname(ass) or ".")


def write_page(cues: list[dict], template: str, video_name: str, out: str) -> None:
    html = open(template, encoding="utf-8").read()
    if "__CUES__" not in html:
        raise SystemExit("템플릿에 __CUES__ 자리가 없다: %s" % template)
    html = html.replace("__CUES__", json.dumps(cues, ensure_ascii=False))
    html = html.replace("__VIDEO__", video_name)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(html)


def main() -> None:
    global FF
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="실행 녹화 → 해설 영상")
    parser.add_argument("--video", required=True, help="녹화 원본 mp4")
    parser.add_argument("--cues", required=True, help="대본 파이썬 파일")
    parser.add_argument("--name", required=True, help="결과 파일 앞에 붙일 이름")
    parser.add_argument("--out", required=True, help="결과를 모아 둘 폴더")
    parser.add_argument("--work", default=None, help="중간 파일 폴더 (기본: 결과 폴더/_build)")
    parser.add_argument("--template", default=os.path.join(here, "page.html"))
    parser.add_argument("--recut", action="store_true", help="구간을 전부 다시 자른다")
    args = parser.parse_args()

    FF = ffmpeg_path()
    segments = load_segments(args.cues)
    work = args.work or os.path.join(args.out, "_build")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(work, exist_ok=True)

    print("구간 자르기 (%d개)" % len(segments))
    parts = cut(args.video, segments, os.path.join(work, "seg"), args.recut)

    print("이어 붙이기")
    mid = os.path.join(work, "mid.mp4")
    concat(parts, os.path.join(work, "seg"), mid)

    cues, total = timeline(segments, parts)
    print("총 길이 %.1f초 (%d:%02d) — 원본 %.1f초"
          % (total, int(total // 60), int(total) % 60, duration(args.video)))

    srt = os.path.join(args.out, "%s_자막.srt" % args.name)
    write_srt(cues, srt)
    write_vtt(cues, os.path.join(work, "subs.vtt"))
    ass = os.path.join(work, "subs.ass")
    write_ass(cues, ass)

    clean_name = "%s_영상.mp4" % args.name
    print("자막 없는 판 인코딩")
    encode(mid, os.path.join(args.out, clean_name))
    print("자막 굽기")
    burn(mid, ass, os.path.join(args.out, "%s_자막포함.mp4" % args.name))

    write_page(cues, args.template, clean_name,
               os.path.join(args.out, "%s_해설.html" % args.name))
    print("완료 — %s" % args.out)


if __name__ == "__main__":
    main()
