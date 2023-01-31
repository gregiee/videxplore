import os
import traceback

import json
import numpy as np
import torch

from demos import MOSEI_sentiment_model, MOSEI_emotion_model
from model.data.datasets.rawvideo_utils import load_audio, load_video

input_dir = "<your_clip_root>"
output_dir = "<your_emo_root>"

# input_dir = "../RTS_clips"
# output_dir = "../RTS_emos"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_emo = MOSEI_emotion_model().to("cuda")
model_sen = MOSEI_sentiment_model().to("cuda")

for folder in os.listdir(f"{input_dir}/videos"):
    emos = {}
    print(f">>> Processing {folder}", flush=True)

    for clip in os.listdir(f"{input_dir}/videos/{folder}"):
        if clip.endswith(".mp4"):
            video_path = f"{input_dir}/videos/{folder}/{clip}"
            audio_path = f"{input_dir}/audios/{folder}/{clip[:-1]}3"

            try:
                video = load_video(video_path, num_frames=8).to("cuda")
                audio = load_audio(audio_path, sr=44100).to("cuda")

                with torch.no_grad():
                    encoder_last_hidden_outputs, *_ = model_emo(video=video, audio=audio)
                    s = model_emo.classifier(encoder_last_hidden_outputs).squeeze().data.cpu().numpy()
                    s = np.round(s, 3)

                with torch.no_grad():
                    encoder_last_hidden_outputs, *_ = model_sen(video=video, audio=audio)
                    sentiment_score = model_sen.classifier(encoder_last_hidden_outputs).squeeze().data.cpu().numpy()

                emos[clip] = {"emo": s.tolist(), "sen": float(sentiment_score.item())}

            except Exception:
                traceback.print_exc()
                print(f">>> Error {clip}", flush=True)

    print(f">>> Finish {folder}", flush=True)
    with open(f"{output_dir}/{folder}.json", "w") as f:
        json.dump(emos, f)
