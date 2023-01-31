import heapq
import os
import random
import traceback
import warnings

import json
from parrot import Parrot

warnings.filterwarnings("ignore")

input_dir = "<emo_caption_root>"
output_dir = "<para_emo_caption_root>"

# input_dir = "./emo_cap"
# output_dir = "./para_emo_cap"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

for caption_file in os.listdir(input_dir):
    try:
        result = {}
        with open(f"{input_dir}/{caption_file}", "r") as f:
            data = json.load(f)["results"]

        for video_id in data:
            new_phrase = []
            captions = {caption["sentence"] for caption in data[video_id]}
            for phrase in captions:
                para_phrases = parrot.augment(input_phrase=phrase, max_return_phrases=2, max_length=64,
                                              adequacy_threshold=0.85, fluency_threshold=0.85,
                                              use_gpu=True)
                if para_phrases is not None:
                    for i in para_phrases:
                        new_phrase.append(i)
            result[video_id] = new_phrase

        print(f">>> Finish {caption_file} (new)", flush=True)
        with open(f"{output_dir}/{caption_file}", "w") as f:
            json.dump(result, f)

    except Exception:
        traceback.print_exc()
        print(f">>> [Error] {caption_file}")
