# RTS archive

A brief guide to processing the RTS archive.

## 0. Collect the video and audio files

**Write scripts to collect video & audio files.**

**How RTS files are organized:** The original video and audio files in the RTS archive are organized as follows:

```sh
ZB000000
├── ZB000000_track1.mp4
├── ZB000000_track2.mp4
├── ZB000000_track3.mp4
├── ...
└── ZB000000_trackn.mp4
```

Video file ends with "track1". Audio file is the largest file among "track2" - "trackn".

**Collect and rename files:** You should write your own scripts to collect and rename the video and audio files. They should be organized as follows:

```sh
<your_rts_root>
├── videos
│    ├── ZB000000.mp4
│    ├── ZB000001.mp4
│    ├── ...
│    └── ZB267218.mp4
└── audios
     ├── ZB000000.mp4
     ├── ZB000001.mp4
     ├── ...
     └── ZB267218.mp4
```

## 0.1 Compress the videos (optional)

**This part will compress FPS and SIZE of videos.**

Requirements:

+ Install [FFmpeg](https://ffmpeg.org/download.html).
+ Install Python libs:

     ``` sh
     pip install ffmpeg-python
     pip install psutil
     ```

Then run:

```bash
python ./00_compress_videos.py --input_root [raw_video_root] --output_root [compressed_video_root]
```

Please not that the directory structure should remain the same as in [0. Collect the video and audio files](#0-collect-the-video-and-audio-files):

```sh
<your_rts_root>
├── videos
│    ├── ZB000000.mp4
│    ├── ZB000001.mp4
│    ├── ...
│    └── ZB267218.mp4
└── audios
     ├── ZB000000.mp4
     ├── ZB000001.mp4
     ├── ...
     └── ZB267218.mp4
```

## 1. Segment the videos into short clips

**This part will segment origianl videos into shorter clips based on detected shots.**

Requirements:

``` sh
pip install moviepy
pip install scenedetect[opencv] --upgrade
```

Set the `input_dir` and `output_dir` variables in `./01_segment_video.py`：

```python
input_dir = "<your_rts_root>"  # Directory root in "0. Collect the video and audio files" or "0.1 Compress the videos".
output_dir = "<your_clip_root>"
```

Then run：

```sh
python ./01_segment_videos.py
```

Segmented clips will be organized in output dir as follows:

```sh
<your_clip_root>
├── videos
│   ├── ZB000000
│   │   ├── ZB000000_00.mp4
│   │   ├── ZB000000_01.mp4
│   │   ├── ...
│   │   └── ZB000000_12.mp4
│   ├── ZB000001
│   │   ├── ZB000001_0.mp4
│   │   ├── ZB000001_1.mp4
│   │   ├── ...
│   │   └── ZB000001_7.mp4
│   └── ...
└── audios
    ├── ZB000000
    │   ├── ZB000000_00.mp3
    │   ├── ZB000000_01.mp3
    │   ├── ...
    │   └── ZB000000_12.mp3
    ├── ZB000001
    │   ├── ZB000001_0.mp3
    │   ├── ZB000001_1.mp3
    │   ├── ...
    │   └── ZB000001_7.mp3
    └── ...
```

## 2. Extract emotions

**This part exacts emotions from segmented clips using out-of-the-box model.**

Requirements:

+ **Install PyTorch=1.10.0 torchvision=0.11.0 torchaudio=0.10.0**
  
+ Install Python libs:

     ```sh
     cd 02_emo_TVLT
     pip install -r requirements.txt
     ```

Set the `input_dir` and `output_dir` variables in `./02_emo.py`：

```python
input_dir = "<your_clip_root>" # Output dir from "1. Segment the videos into short clips"
output_dir = "<your_emo_root>"
```

Then run:

```sh
python ./02_emo.py
cd ..
```

Output files are JSON files for each orginal video. Within each JSON file, is the extracted emotions for each clip segmented (For example, if  `ZB000001.mp4` is segmented into 7 clips, then 7 emotions of 7 clips will be all stored in `ZB000001.json`). Output JSONs are organized as follows:

```sh
<your_emo_root>
├── ZB000000.json
├── ZB000001.json
├── ZB000002.json
├── ...
└── ZB279854.json
```

## 3. Generate video descriptions

**This part use out-of-box-model to generate text discriptions for segmented video clips.**

Requirements:

```sh
cd 03_cap_PDVC
pip install -r requirement.txt

cd pdvc/ops
sh make.sh
cd ../..
```

Set the `CLIP_FOLDER` and `OUTPUT_FOLDER` in `03_cap.sh`:

```sh
CLIP_FOLDER=<your_clip_root> # Output dir from "1. Segment the videos into short clips"
OUTPUT_FOLDER=<naive_caption_root>
```

Then run:

```sh
sh 03_cap.sh
cd ..
```

This step will generate JSON files for each original video, with video descriptions for each clips segmented. Output JSONs are organized as follows:

```sh
<naive_caption_root>
├── ZB000000.json
├── ZB000001.json
├── ZB000002.json
├── ...
└── ZB279854.json
```

## 4. Fuse and paraphrase descriptions

**This part generate two dataset of video clips and text description pairs,**

+ **one with emotion - emotional dataset (step 1 & 2),**
  + **Video descriptions in the emotional dataset are called *emotional descriptions*.**
  
+ **one without - naive dataset (step 3).**
  + **Video descriptions in the naive dataset are called *naive descriptions*. [Part 3. Generate video descriptions](#3-generate-video-descriptions) already generates some naive descriptions.**

Requirements:

```sh
pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git
```

**Step 1: generate hard-coded emotional descriptions.**

Set the three variables in `04_fuse.py`:

```python
input_dir = "<naive_caption_root>" # Output dir from "3. Generate video descriptions"
emo_dir = "<your_emo_root>" # Output dir from "2. Extract emotions"
output_dir = "<emo_caption_root>"
```

Then run:

```sh
python ./04_fuse.py
```

Generated folder structure:

```sh
<emo_caption_root>
├── ZB000000.json
├── ZB000001.json
├── ZB000002.json
├── ...
└── ZB279854.json
```

**Step 2: paraphrase hard-coded emotional descriptions.**

This step will paraphrase outcome from **Step 1**, to enhance the hard-coded emotional descriptions.

Set the two variables in `04_paraphrase_emo.py`:

```python
input_dir = "<emo_caption_root>" # Output dir from Step 1
output_dir = "<para_emo_caption_root>"
```

Then run:

```sh
python ./04_paraphrase_emo.py
```

```sh
<para_emo_caption_root>
├── ZB000000.json
├── ZB000001.json
├── ZB000002.json
├── ...
└── ZB279854.json
```

**Step 3: paraphrase naive descriptions.**

This step is parallel to **Step 1 & 2**, the aim is to paraphrase the outcome from [Part 4. Generate video descriptions](#4-fuse-and-paraphrase-descriptions).

Set the three variables in `04_paraphrase.py`:

```python
input_dir = "<naive_caption_root>"  # Output dir from "3. Generate video descriptions"
output_dir = "<para_naive_caption_root>"
```

Then run:

```sh
python ./04_paraphrase.py
```

```sh
<para_naive_caption_root>
├── ZB000000.json
├── ZB000001.json
├── ZB000002.json
├── ...
└── ZB279854.json
```

## 5. Sample and format the descriptions

**This part will format the text description and video clips pairs to the format of selected text-video retrival model's standard.**

Set the following variables in `05_build_dataset.py`:

```python
emo_dir = "<your_emo_root>"  # Output dir from "2. Extract emotions"
caption_dir = "<naive_caption_root>" # Output dir from "3. Generate video descriptions"
new_caption_dir = "<para_naive_caption_root>" # Output dir from "4. Fuse and paraphrase descriptions - step 3"
emo_caption_dir = "<emo_caption_root>" # Output dir from "4. Fuse and paraphrase descriptions - step 1"
new_emo_caption_dir = "<para_emo_caption_root>" # Output dir from "4. Fuse and paraphrase descriptions - step 2"

train_set_ratio = 0.9   # Train / Val split ratio
output_dir = "<your_annotation_root>"
```

Then run:

```sh
python ./05_build_dataset.py
```

This will filter and format descriptions and save them to files:

```sh
<your_annotation_root>
├── dataset.csv # list of all videos
├── train.csv   # video list for training
├── naive_captions.json     # naive descriptions
├── emotional_captions.json # emotional descriptions
├── naive_test.csv      # videos and naive descriptions for validation
└── emotional_test.csv  # videos and emotional descriptions for validation
```

## 6. Fine-tune text-to-video retrieval models

**This part fine tunes the pretrained model for text-to-video retrieval with prepared two datasets.**

Requirements:

+ **Important:** Starting from this part, we suggest creating a new Python environment. Then install **pytorch=1.7.1 torchvision=0.8.2** and do **all the following parts** in the new env to avoid potential issues.

+ Install some Python libs:
  
     ```sh
     cd 06_retrieval_ts2 
     pip install ftfy regex tqdm
     pip install opencv-python boto3 requests pandas scipy
     ```

Set the following paths in `train_naive_model.sh` and `train_emotional_model.sh`:

```sh
CLIP_ROOT=<your_clip_root>  # Ourput dir from "1. Segment the videos into short clips"
ANNOTATION_ROOT=<your_annotation_root> # Output dir from "5. Sample and format the descriptions"
OUTPUT_ROOT=<your_model_root> 
```

You can also adjust the training parameters in both scripts.

To fine-tune model on the naive dataset, run:

```sh
sh ./train_naive_model.sh
cd ..
```

To fine-tune model on the emotional dataset, run:

```sh
sh ./train_emotional_model.sh
cd ..
```

The fine-tuned models will be saved in `<your_model_root>/naive/` or `<your_model_root>/emotinal/` based on the dataset they used. You can find a log file in the both folders for evaluation results and the best models.

## 7. Encode videos or descriptions

**This part will encode videos (from a folder) or descriptions (from a text file) to high-dimensional vectors via the selected model.**

Set the six variables in `06_retrieval_ts2/07_get_emb.py` `line 343`:

```python
input_type = "video"                   # "video" or "text"
text_file = "<your_description_file>"  # Only used if input_type == "text"
video_dir = "<your_video_root>"        # Only used if input_type == "video"
model_path = "<your_model_file>"       # E.g., <your_model_root>/naive/pytorch_model.bin.1
output_path = "<your_output_npy_file>"  # E.g., ./embedding.npy
video_order_file = "<your_video_order_file>"  # E.g., ./video_order.txt. Save the list of videos to file. Only used if input_type == "video"
```

If you feel lost, here are detailed explanations for these variables:

+ If `input_type` is `video`, the model (`<your_model_file>`) will encode all videos in the folder (`<your_video_root>`) and save the obtained vectors into a `.npy` file (`<your_output_npy_file>`). The order of encoded videos is saved in `<your_video_order_file>`. Note that the order of videos (in `<your_video_order_file>`) matches the order of vectors (in `<your_output_npy_file>`) for further utilization.

+ If `input_type` is `text`, the model (`<your_model_file>`) will encode all descriptions in the file (`<your_description_file>`) and save the obtained vectors into a `.npy` file (`<your_output_npy_file>`). The order of vectors (in `<your_output_npy_file>`) matches the order of descriptions (in `<your_description_file>`).

Then run:

```sh
cd 06_retrieval_ts2
python ./07_get_emb.py
cd ..
```

## 8. Reduce dimension

**This part will reduce high-dimensional vectors generated in the previous step to low-dimensional vectors and save them to a file (`<your_point_file>`). Then the file can be loaded as coordinates of points and visualized as you wish.**

Requirements:

```sh
pip install umap-learn
```

Set the following variables and UMAP parameters in `08_prepare_viz.py`:

```python
emb = np.load("<your_output_npy_file>")  # Generated .npy file in "7. Encode videos or descriptions"
point_file = "<your_point_file>"         # E.g., ./points.txt
n_components = 2    # Target dimensionality. 2 for generating 2d points, 3 for generating 3d points...

# See https://umap-learn.readthedocs.io/en/latest/ for more details.
mapper = umap.UMAP(n_neighbors=8,
                   min_dist=0.5,
                   n_components=n_components,
                   metric='euclidean')
```

Then run:

```sh
python ./08_prepare_viz.py
```

The low-dimentional vectors will be saved in `<your_point_file>`. Each vector is saved in a line. Components are separated by `,`.

## 9. A viz demo (Optional)

We also provide a simple Unity scene to show how `<your_point_file>`, `<your_description_file>`, and `<your_video_order_file>` can be used. We test it on Unity 2021.3.15.

To visualize the points in Unity project:

+ Copy `<your_point_file>`generated in the previous step to `Assets/Resources/Texts`.
  
+ If you want to display videos, copy the `<your_video_order_file>` to `Assets/Resources/Texts`. And copy all related video files to `Assets/Resources/Videos`.

+ If you want to display text descriptions, copy the `<your_description_file>` to `Assets/Resources/Texts`.

The Unity project structure:

```sh
Assets
├── Resources
│    ├── Texts
│    │   ├── <your_point_file>
│    │   ├── <your_video_order_file> 
│    │   └── <your_description_file>
│    └── Videos
│        ├──  xxxxxx.mp4
│        ├──  yyyyyy.mp4
│        └──  ...
└── ...
```

In the sample scene, use WASD to adjust viewport and move mouse to cubes to display more information.

If you generate `<your_point_file>`, `<your_description_file>`, or `<your_video_order_file>` in Unix, You may need to replace the characters `"\r\n"` (in `Assets/Scripts/PointParser.cs` `line 29 - 31`) with `"\n"`.

```C++
string[] pointText = pointFile.text.Split("\r\n");
string[] videoPathsText = videoPathsFile.text.Split("\r\n");
string[] descriptionText = descriptionFile.text.Split("\r\n");
```
