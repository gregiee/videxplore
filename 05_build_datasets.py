import heapq
import os
import random

import json
import pandas as pd
from tqdm import tqdm

emo_dir = "<your_emo_root>"
caption_dir = "<naive_caption_root>"
new_caption_dir = "<para_naive_caption_root>"
emo_caption_dir = "<emo_caption_root>"
new_emo_caption_dir = "<para_emo_caption_root>"
train_set_ratio = 0.9
output_dir = "<your_annotation_root>"

# emo_dir = "./RTS_emos"
# caption_dir = "./naive_cap"
# new_caption_dir = "./para_naive_cap"
# emo_caption_dir = "./emo_cap"
# new_emo_caption_dir = "./para_emo_cap"
# train_set_ratio = 0.9
# output_dir = "./RTS_annos"


def read_captions(video_id, original_dir, phrased_dir):
    filename = f"{video_id.split('_')[0]}"

    with open(f"{original_dir}/{filename}.json", "r") as f:
        orig_data = json.load(f)["results"][video_id]
    orig_captions = {caption["sentence"] for caption in orig_data}

    with open(f"{phrased_dir}/{filename}.json", "r") as f:
        data = json.load(f)
        if video_id in data:
            para_data = data[video_id]
            para_captions = {caption[0] for caption in para_data if caption[1] > 10}
        else:
            para_captions = set()

    return orig_captions | para_captions


def generate_datasets(video_set,
                      original_dir, phrased_dir, output_path,
                      original_emo_dir, phrased_emo_dir, emo_output_path):
    results = []
    emo_results = []

    caption_key = "caption"
    video_key = "video_id"
    entry_key = "sentences"

    for video_id in tqdm(video_set):
        captions = read_captions(video_id, original_dir, phrased_dir)
        emo_captions = read_captions(video_id, original_emo_dir, phrased_emo_dir)

        min_length = min(len(captions), len(emo_captions))
        for sentence in random.sample(captions, min_length):
            results.append({caption_key: sentence, video_key: video_id})
        for sentence in random.sample(emo_captions, min_length):
            emo_results.append({caption_key: sentence, video_key: video_id})

    with open(output_path, "w") as f:
        json.dump({entry_key: results}, f)
    with open(emo_output_path, "w") as f:
        json.dump({entry_key: emo_results}, f)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Sample training set based on split ratio.
emo_counts = [set() for i in range(0, 6)]
for file in os.listdir(emo_dir):
    if not file.endswith("json"):
        continue
    with open(f"{emo_dir}/{file}") as f:
        data = json.load(f)
    for video in data:
        el = data[video]['emo']
        largest = el.index(max(el))
        if largest == 0:
            second = el.index(heapq.nlargest(2, el)[1])
            if second == 1 or second == 0:
                emo_counts[0].add(video[:-4])
            else:
                emo_counts[second].add(video[:-4])
        else:
            emo_counts[largest].add(video[:-4])

selected_count = [len(i) for i in emo_counts]
train_count = [int(i * train_set_ratio) for i in selected_count]
val_count = [selected_count[i] - train_count[i] for i in range(0, 6)]

selected_set = [set(random.sample(emo_counts[i], selected_count[i])) for i in range(0, 6)]
train_set = [set(random.sample(selected_set[i], train_count[i])) for i in range(0, 6)]
val_set = [selected_set[i] - train_set[i] for i in range(0, 6)]

selected_df = []
for i in range(0, 6):
    for j in train_set[i]:
        selected_df.append([j, i, "train"])
    for j in val_set[i]:
        selected_df.append([j, i, "val"])
selected_df = pd.DataFrame(selected_df, columns=['video_id', 'emo_id', 'split'])
selected_df.to_csv(f"{output_dir}/dataset.csv", index=False)

# Generate standard training sets for "naive captions" and "emotional captions"
all_set = selected_set[0] | selected_set[1] | selected_set[2] | selected_set[3] | selected_set[4] | selected_set[5]
generate_datasets(all_set,
                  caption_dir, new_caption_dir, f"{output_dir}/naive_captions.json",
                  emo_caption_dir, new_emo_caption_dir, f"{output_dir}/emotional_captions.json")

# Save the list of training set.
all_train_set = set()
for i in train_set:
    all_train_set = all_train_set | i
train_df = pd.DataFrame(list(all_train_set), columns=["video_id"])
train_df.to_csv(f"{output_dir}/train.csv", index=False)

all_test_set = set()
for i in val_set:
    all_test_set = all_test_set | i

# Save the list of validation set for "naive dataset"
with open(f"{output_dir}/naive_captions.json", "r") as f:
    data = json.load(f)["sentences"]
data_dict = {}
for i in data:
    data_dict[i["video_id"]] = i["caption"]
test_df = [[i, data_dict[i]] for i in all_test_set]
test_df = pd.DataFrame(test_df, columns=["video_id", "sentence"])
test_df.to_csv(f"{output_dir}/naive_test.csv", index=False)

# Save the list of validation set for "emotional dataset"
with open(f"{output_dir}/emotional_captions.json", "r") as f:
    data = json.load(f)["sentences"]
data_dict = {}
for i in data:
    data_dict[i["video_id"]] = i["caption"]
test_df = [[i, data_dict[i]] for i in all_test_set]
test_df = pd.DataFrame(test_df, columns=["video_id", "sentence"])
test_df.to_csv(f"{output_dir}/emotional_test.csv", index=False)
