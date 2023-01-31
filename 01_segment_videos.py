import math
import os
import subprocess
import traceback

import moviepy.editor as mp
from scenedetect import detect, AdaptiveDetector

input_dir = "<your_rts_root>"
output_dir = "<your_clip_root>"

# input_dir = "./RTS_data"
# output_dir = "./RTS_clips"

for filename in os.listdir(f"{input_dir}/videos"):
    print(f">>> Processing {filename}...", flush=True)

    video_file = f"{input_dir}/videos/{filename}"
    audio_file = f"{input_dir}/audios/{filename}"

    try:
        # scene_list = detect(video_file, AdaptiveDetector(luma_only=True), stats_file_path="./01_segment_stats.csv")
        scene_list = detect(video_file, AdaptiveDetector(luma_only=True))

        resemble_scenes = []
        st = 0
        for i in range(len(scene_list)):
            duration = scene_list[i][1] - scene_list[st][0]
            if duration.get_seconds() >= 12:
                resemble_scenes.append((scene_list[st][0], scene_list[i][1]))
                st = i + 1
        if st < len(scene_list):
            resemble_scenes[-1] = (resemble_scenes[-1][0], scene_list[-1][1])

        # "ffmpeg -v error -nostdin -y -ss <start_time> -i <input_video_path> -t <duration>
        # -c:v libx264 -preset fast -crf 23 -c:a aac -sn <name>"
        # print(f">>> Cutting {filename}...", flush=True)
        if len(resemble_scenes) <= 10:
            padding = 1
        else:
            padding = int(math.log10(len(resemble_scenes) - 1) + 1)
        cmd_str = "ffmpeg -v error -nostdin -y -ss %.3f -i %s -t %.3f -c:v libx264 -preset fast -crf 23 -c:a aac -sn %s"

        video_id = filename[:-4]
        video_folder = f"{output_dir}/videos/{video_id}"
        audio_folder = f"{output_dir}/audios/{video_id}"
        
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
            
        for i, (start_time, end_time) in enumerate(resemble_scenes):
            duration = end_time - start_time
            subprocess.call(cmd_str % (start_time.get_seconds(), video_file, duration.get_seconds(), f"{video_folder}/{video_id}_{i:0{padding}d}.mp4"), shell=True)
        # print(">>> Finish Cutting Video", flush=True)

        for i, (start_time, end_time) in enumerate(resemble_scenes):
            duration = end_time - start_time
            tmp_file = f"{audio_folder}/{video_id}_{i:0{padding}d}.mp4"
            subprocess.call(cmd_str % (start_time.get_seconds(), audio_file, duration.get_seconds(), tmp_file), shell=True)
            mp.AudioFileClip(tmp_file).write_audiofile(f"{audio_folder}/{video_id}_{i:0{padding}d}.mp3")
            os.remove(tmp_file)

        # print(">>> Finish Cutting Audio", flush=True)
        # os.remove(video_file)
        # os.remove(audio_file)
        print(f">>> Finish {video_id}", flush=True)
        
    except Exception:
        traceback.print_exc()
        print(f">>> [Error] {filename}")