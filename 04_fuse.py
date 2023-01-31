import heapq
import os
import random
import traceback
import warnings

import json

input_dir = "<naive_caption_root>"
emo_dir = "<your_emo_root>"
output_dir = "<emo_caption_root>"

# input_dir = "./naive_cap"
# emo_dir = "./RTS_emos"
# output_dir = "./emo_cap"


def emotion_paraphrase(sentence, emotion):
    anger_morph = ["in anger", "with anger", "angrily", "in annoyance", "with annoyance", "with hate", "in disapproval"]
    disgust_morph = ["in disgust", "with disgust", "disgustedly"]
    happy_morph = ["with joy", "joyously", "joyfully", "in amusement", "with amusement",
                   "with excitement", "in excitement", "excitedly", "with relief", "with happiness",
                   "happily", "with enthusiasm", "enthusiastically"]
    sad_morph = ["with sadness", "sadly", "in disappointment", "disappointedly", "with grief",
                 "in grief", "pessimistically"]
    fear_morph = ["in fear", "with fear", "out of fear", "fearfully", "from nervousness",
                  "out of nervousness", "nervously", "with worry", "worriedly", "confusedly"]
    surprise_morph = ["in surprise", "with surprise", "surprisedly", "with curiosity", "curiously"]
    if emotion == 0:
        return f"{sentence[:-1]} {random.sample(happy_morph, 1)[0]}."
    elif emotion == 1:
        return f"{sentence[:-1]} {random.sample(sad_morph, 1)[0]}."
    elif emotion == 2:
        return f"{sentence[:-1]} {random.sample(anger_morph, 1)[0]}."
    elif emotion == 3:
        return f"{sentence[:-1]} {random.sample(fear_morph, 1)[0]}."
    elif emotion == 4:
        return f"{sentence[:-1]} {random.sample(disgust_morph, 1)[0]}."
    elif emotion == 5:
        return f"{sentence[:-1]} {random.sample(surprise_morph, 1)[0]}."
    else:
        print(">>> [ERROR]")
    return


def find_emo(emos, video_id):
    # print(emos)
    # print(video_id)
    el = emos[f"{video_id}.mp4"]['emo']
    largest = el.index(max(el))
    if largest == 0:
        second = el.index(heapq.nlargest(2, el)[1])
        if second == 1:
            return 0
        else:
            return second
    else:
        return largest


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for caption_file in os.listdir(input_dir):
        try:
            results = {"results": {}}
            with open(f"{input_dir}/{caption_file}", "r") as f:
                data = json.load(f)["results"]
            with open(f"{emo_dir}/{caption_file}", "r") as f:
                emos = json.load(f)

            for video_id in data:
                new_phrase = []
                emo = find_emo(emos, video_id)
                captions = {caption["sentence"] for caption in data[video_id]}
                for phrase in captions:
                    new_phrase.append({"sentence": emotion_paraphrase(phrase, emo)})
                results["results"][video_id] = new_phrase

            print(f">>> Finish {caption_file}", flush=True)
            with open(f"{output_dir}/{caption_file}", "w") as f:
                json.dump(results, f)

        except Exception:
            traceback.print_exc()
            print(f">>> [Error] {caption_file}")
            raise
