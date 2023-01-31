import argparse
import datetime
import os
import random
import time

import torch
import numpy as np
import pandas as pd

from metrics import compute_metrics
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4Clip
from util import parallel_apply
from video_extractor import RawVideoExtractorCV2 as RawVideoExtractor


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_eval", action='store_true', default=True, help="Whether to run eval on the dev set.")

    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    # parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', default=True,
                        help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', default=True)

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="seqTransf",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--best_ckpt_path", default="", type=str, help="Choose a ckpt to use")
    parser.add_argument("--eval_in_train", action='store_true', help="Whether to do eval in training.")

    parser.add_argument('--train_csv', type=str, default='./izar/msrvtt_data/MSRVTT_train.9k.csv', help='')
    parser.add_argument('--val_csv', type=str, default='./application/MSRVTT_JSFUSION_test_sample.csv', help='')
    parser.add_argument('--data_path', type=str, default='./izar/msrvtt_data/MSRVTT_data.json',
                        help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='./application/test_1k_compress', help='feature path')
    args = parser.parse_args()

    # # Check paramenters
    # if args.gradient_accumulation_steps < 1:
    #     raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
    #         args.gradient_accumulation_steps))
    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)
    return args


def find_top_n_idx(x, n):
    return x.argsort()[-n:][::-1]


def get_batch_text_embeddings(sentences, max_words=32):
    k = len(sentences)
    pairs_text = np.zeros((k, 1, max_words), dtype=np.compat.long)
    pairs_mask = np.zeros((k, 1, max_words), dtype=np.compat.long)
    pairs_segment = np.zeros((k, 1, max_words), dtype=np.compat.long)

    for i, sentence in enumerate(sentences):
        words = tokenizer.tokenize(sentence)

        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_words
        assert len(input_mask) == max_words
        assert len(segment_ids) == max_words

        pairs_text[i] = np.array(input_ids, dtype=np.compat.long)
        pairs_mask[i] = np.array(input_mask, dtype=np.compat.long)
        pairs_segment[i] = np.array(segment_ids, dtype=np.compat.long)

    pairs_text = torch.tensor(pairs_text).to(device).detach()
    pairs_mask = torch.tensor(pairs_mask).to(device).detach()
    pairs_segment = torch.tensor(pairs_segment).to(device).detach()

    return pairs_text, pairs_mask, pairs_segment


def get_batch_video_embeddings(choice_video_ids):
    max_frames = 12
    slice_framepos = 2
    frame_order = 0
    rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)

    video_mask = np.zeros((len(choice_video_ids), 1, max_frames), dtype=np.compat.long)
    max_video_length = [0] * len(choice_video_ids)

    # Pair x L x T x 3 x H x W
    video = np.zeros((len(choice_video_ids), 1, max_frames, 1, 3,
                      rawVideoExtractor.size, rawVideoExtractor.size), dtype=float)

    for i, video_path in enumerate(choice_video_ids):
        # video_path = f"{features_path}/{video_id}.mp4"
        if os.path.exists(video_path) is False:
            raise

        raw_video_data = rawVideoExtractor.get_video_data(video_path)
        raw_video_data = raw_video_data['video']
        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = rawVideoExtractor.process_raw_data(raw_video_data_clip)
            if max_frames < raw_video_slice.shape[0]:
                if slice_framepos == 0:
                    video_slice = raw_video_slice[:max_frames, ...]
                elif slice_framepos == 1:
                    video_slice = raw_video_slice[-max_frames:, ...]
                else:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            video_slice = rawVideoExtractor.process_frame_order(video_slice, frame_order=frame_order)

            slice_len = video_slice.shape[0]
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[i][:slice_len, ...] = video_slice
        else:
            print("video path: {} error.".format(video_path))

    for i, v_length in enumerate(max_video_length):
        video_mask[i][:v_length] = [1] * v_length

    return torch.tensor(video).to(device).detach(), torch.tensor(video_mask).to(device).detach()


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits = model.get_final_similarity(sequence_output, visual_output, input_mask, video_mask,
                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(model, test_dataloader, device, n_gpu=1):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()

    with torch.no_grad():
        batch_list_t, batch_list_v = [], []
        batch_sequence_output_list, batch_visual_output_list = [], []

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask,
                                                                              video, video_mask)

            batch_sequence_output_list.append(sequence_output)
            batch_list_t.append((input_mask, segment_ids,))
            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask,))
            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu < 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in
                                     device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list,
                                            batch_visual_output_list)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    print("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    tv_metrics = compute_metrics(sim_matrix)
    print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    print("------------------------------------------------------------")
    print("Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
          format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'],
                 tv_metrics['MeanR']))
    R1 = tv_metrics['R1']
    return R1


def load_ts2_net(file_path):
    args = get_args()
    model_state_dict = torch.load(file_path, map_location='cpu')
    model = CLIP4Clip.from_pretrained("cross-base",
                                      cache_dir="",
                                      state_dict=model_state_dict,
                                      task_config=args)

    model.to(device)
    model.eval()
    return model


# def get_video_list(anno_dir, emo_id=-1):
#     # anno_dir = "../izar/RTS_annos"
#     df = pd.read_csv(f'{anno_dir}/datasets.csv')
#     df = df[df['split'] == 'val']
#     if emo_id == -1:
#         return df['video_id'].tolist()
#
#     val_df = df[df['emo_id'] == emo_id]
#     return val_df['video_id'].tolist()


def get_video_list(anno_root):
    df = pd.read_csv(f'{anno_root}/rts_emo_test.csv')
    return df['video_id'].tolist()


def get_text_list(anno_root):
    df = pd.read_csv(f'{anno_root}/rts_emo_test.csv')
    return df['sentence'].tolist()


if __name__ == "__main__":
    batch_size = 20

    input_type = "video"  # "video" or "text"
    text_file = "<your_description_file>"  # Only used if input_type is "text"
    video_dir = "<your_video_root>"  # Only used if input_type is "video"
    model_path = "<your_model_file>"  # E.g., <your_output_root>/naive/pytorch_model.bin.1
    output_path = "<your_output_npy_file>"  # E.g., ./embedding.npy
    video_order_file = "<your_video_order_file>"  # E.g., ./video_order.txt. Save the list of videos to file.
    
    # input_type = "video"                    # "video" or "text"
    # text_file = "<your_description_file>"   # Only used if input_type is "text"
    # video_dir = "../RTS_clips/videos/ZB000000"         # Only used if input_type is "video"
    # model_path = "../RTS_models/emotional/pytorch_model.bin.2"        # E.g., <your_output_root>/naive/pytorch_model.bin.1
    # output_path = "../test.npy"  # E.g., ./embedding.npy
    # video_order_file = "../test_video_order.txt" # E.g., ./video_order.txt. Save the list of videos to file.

    # batch_text = get_text_list(anno_dir)
    # batch_videos = get_video_list(anno_dir)
    batch_text, batch_videos = [], []
    if input_type == "text":
        with open(text_file, "r") as f:
            batch_text = [line.rstrip() for line in f]
    elif input_type == "video":
        batch_videos = [f"{video_dir}/{i}" for i in os.listdir(video_dir)]
        with open(video_order_file, "w") as f:
            for i in range(len(batch_videos)):
                f.write(f"{batch_videos[i]}")
                if i != len(batch_videos) - 1:
                    f.write("\n")

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                     "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    tokenizer = ClipTokenizer()

    # video_dir = "../izar/RTS_clips"

    # if model_mode == "emo":
    #     model_path = "../izar/models/ts2_emo.bin.1"
    # else:
    #     model_path = "../izar/models/ts2_ori.bin.1"
    model = load_ts2_net(model_path)

    start_idx = 0
    end_idx = batch_size

    batch_list_t, batch_list_v = [], []
    batch_sequence_output_list, batch_visual_output_list = [], []

    with torch.no_grad():
        remainder = len(batch_videos) % batch_size
        length = len(batch_videos) // batch_size if remainder == 0 else (len(batch_videos) // batch_size) + 1
        for i in range(0, length):
            if input_type == "text":
                input_ids, input_mask, segment_ids = get_batch_text_embeddings(batch_text[start_idx:end_idx])
                sequence_output = model.get_sequence_output_emb(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)

            elif input_type == "video":
                video, video_mask = get_batch_video_embeddings(batch_videos[start_idx:end_idx])
                visual_output = model.get_visual_output_emb(video, video_mask)
                batch_visual_output_list.append(visual_output)

            # sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)
            # batch_list_t.append((input_mask, segment_ids,))
            # batch_list_v.append((video_mask,))

            start_idx = end_idx
            end_idx = start_idx + batch_size

        emb = []
        if input_type == "video":
            emb_list = batch_visual_output_list
        else:
            emb_list = batch_sequence_output_list
        for i in emb_list:
            for j in i.cpu().numpy():
                emb.append(j[0])
        emb_np = np.array(emb)
        np.save(output_path, emb_np)
