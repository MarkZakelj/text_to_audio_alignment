"""Test decoders"""
import os
from pathlib import Path
import re
import json
import string
import numpy as np
import gzip

import torch

from ctcdecode import decoders
from ctcdecode.scorer import KenLMScorer
from ctcdecode.tokenizer import CharTokenizer
from ctcdecode.utils import build_fst_from_words_file
from generate_lm import build_lm, convert_and_filter_topk
import argparse
from tqdm import tqdm


def get_slices(array, slice_size):
    n = len(array)
    n_slices = n // slice_size
    sizes = np.full(n_slices, slice_size)
    while np.sum(sizes) < n:
        sizes[:n % slice_size] += 1
    start = 0
    for size in sizes:
        yield array[start:start + size]
        start += size


def min_segment(segments):
    min_idx = 0
    min_size = segments[0][1] - segments[0][0]
    for i, seg in enumerate(segments):
        size = seg[1] - seg[0]
        if size < min_size:
            min_idx = i
            min_size = size
    return segments[min_idx], min_idx


def merge_segments(segments, min_len=10):
    left, right = -1, 1

    def size(x):
        return x[1] - x[0]

    tmp_segments = [e for e in segments]
    min_seg, min_idx = min_segment(tmp_segments)
    while size(min_seg) / 50 < min_len and len(tmp_segments) > 1:
        if min_idx == 0:
            direction = right
        elif min_idx == len(tmp_segments) - 1:
            direction = left
        else:
            left_size = size(tmp_segments[min_idx - 1])
            right_size = size(tmp_segments[min_idx + 1])
            if left_size < right_size:
                direction = left
            else:
                direction = right
        first, second = min_idx, min_idx + direction  # if direction == right
        if direction == left:
            first, second = second, first
        tmp_segments[first] = tmp_segments[first][0], tmp_segments[second][1]
        del tmp_segments[second]
        min_seg, min_idx = min_segment(tmp_segments)
    return tmp_segments


def to_char_based_text(text):
    lower_text = text.lower()
    lower_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    lower_text = lower_text.replace('\n', ' ')
    lower_text = re.sub(' +', '-', lower_text)
    lower_text = ' '.join(list(lower_text))
    return lower_text


# def generate_dashed_text(text, is_song_form=False):
#     if is_song_form:
#         new_lines = []
#         for line in text.split('\n'):
#             new_lines.append(line.replace(' ', '-'))
#     else:


def preprocess_to_word(char_text):
    text = char_text.replace(' ', '').replace('-', ' ')
    return text


def decode_segment(segment, offset, mode, tokenizer, scorer):
    if mode == 'greedy':
        result = decoders.ctc_greedy_decoder(segment, blank=tokenizer.blank_idx)
        best_result = result
    elif mode == 'lm+fst':
        result = decoders.ctc_beam_search_decoder(segment,
                                                  lm_scorer=scorer,
                                                  blank=tokenizer.blank_idx,
                                                  beam_size=200)
        best_result = result[0]
    else:
        raise ValueError('mode must be one of: "greedy" | "lm+fst"')

    if mode == 'lm+fst' and not best_result[1]:
        return [], "", []
    # remove spaces at the end
    best_token_ids = np.trim_zeros(np.array(best_result[1]), 'b')
    new_l = len(best_token_ids)
    frame_ids = np.array(best_result[2])[:new_l]
    frame_ids = np.append(frame_ids, frame_ids[-1])
    frame_ids += offset
    txt_result = ''.join(tokenizer.idx2token(best_result[1]))
    word_starts = [0] + list(frame_ids[np.where(best_token_ids == 0)[0] + 1] * 0.02)
    word_ends = list(frame_ids[np.nonzero(best_token_ids == 0)[0] - 1] * 0.02) + [frame_ids[-1] * 0.02]
    segment_transcription_json = [{'transcript': triple[0], 'start': triple[1], 'end': triple[2]}
                                  for triple in zip(txt_result.split(' '), word_starts, word_ends)]
    return frame_ids.tolist(), txt_result, segment_transcription_json


def real_ctc_decode(filename, data_dir, frame_segments, model_name, mode='lm+fst'):
    data = np.load(os.path.join(data_dir, 'results', model_name, 'output.npy'))

    inputs = torch.as_tensor(data).log_softmax(1)

    tokenizer = CharTokenizer('labels.txt')

    # greedy using beam
    greedy_result = decoders.ctc_greedy_decoder(inputs[frame_segments[0][0]:frame_segments[-1][1], :],
                                                blank=tokenizer.blank_idx)
    txt_result_greedy = ''.join(tokenizer.idx2token(greedy_result[1]))

    print(txt_result_greedy)
    transcription_json = []

    scorer = KenLMScorer(os.path.join(data_dir, 'lm.arpa'),
                         tokenizer,
                         trie_path=os.path.join(data_dir, 'words_trie.fst'),
                         alpha=0.01,
                         beta=0.0,
                         unit='word')
    print('unmerged segments:', frame_segments)
    merged_segments = merge_segments(frame_segments)
    print('merged segments:', merged_segments)

    transcription_json = []
    res = []
    frame_idss = []
    for start, end in tqdm(merged_segments):
        if mode == 'greedy':
            result = decoders.ctc_greedy_decoder(inputs[start:end, :], blank=tokenizer.blank_idx)
            best_result = result
        elif mode == 'lm+fst':
            result = decoders.ctc_beam_search_decoder(inputs[start:end, :],
                                                      lm_scorer=scorer,
                                                      blank=tokenizer.blank_idx,
                                                      beam_size=200)
            best_result = result[0]
        else:
            raise ValueError('mode must be one of: "greedy" | "lm+fst"')
        # result = decoders.ctc_beam_search_decoder(inputs[start:end, :],
        #                                           lm_scorer=scorer,
        #                                           blank=tokenizer.blank_idx,
        #                                           beam_size=200)
        # best_result = result[0]
        if len(best_result[1]) == 0:
            continue
        best_token_ids = np.trim_zeros(np.array(best_result[1]), 'b')
        last_id = best_result[2][-1]
        new_l = len(best_token_ids)
        frame_ids = np.array(best_result[2])[:new_l]
        frame_ids = np.append(frame_ids, last_id)
        frame_ids += start
        frame_idss += list(frame_ids)
        txt_result = ''.join(tokenizer.idx2token(best_result[1]))
        res.append(txt_result)

        word_starts = [0] + list(frame_ids[np.where(best_token_ids == 0)[0] + 1] * 0.02)
        word_ends = list(frame_ids[np.where(best_token_ids == 0)[0] - 1] * 0.02) + [frame_ids[-1] * 0.02]
        transcription_json += [{'transcript': triple[0], 'start': triple[1], 'end': triple[2]}
                               for triple in zip(txt_result.split(' '), word_starts, word_ends)]
    return np.array(frame_idss) * 0.02, ' '.join(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', nargs=1, type=str, help='Primer, ki se bo procesiral. Ce je all, izberi vse',
                        required=True)
    parser.add_argument('-m', '--mode', nargs=1, type=str, help='nacin izvajanja. "greedy" ali "lm+fst"', required=True)
    parser.add_argument('--model', type=str, help='Name of asr model used', required=True)
    args = parser.parse_args()
    sample_name = args.name[0]
    ctc_mode = args.mode[0]
    model_name = args.model.replace('.nemo', '')
    if sample_name != 'all':
        filenames = [sample_name]
    else:
        filenames = [name.name for name in Path('audio_samples/').iterdir()]
    print(filenames)
    for filename in filenames:
        data_dir = os.path.join('audio_samples', filename)

        _steps = {2, 3, 4}

        if 2 in _steps:
            # generate lm
            args = argparse.Namespace()
            args.input_txt = os.path.join(data_dir, 'main_text.txt')
            args.output_dir = data_dir
            args.top_k = 5000
            args.kenlm_bins = os.path.join('kenlm', 'build', 'bin')
            args.arpa_order = 5
            args.max_arpa_memory = "80%"
            args.binary_a_bits = 255
            args.binary_q_bits = 8
            args.binary_type = 'trie'
            args.discount_fallback = True

            data_lower, vocab_str = convert_and_filter_topk(args)
            with gzip.open(data_lower, 'rb') as f:
                origin_text = f.read().decode('utf8')
                print(origin_text)
            origin_text_phrases = [{'transcript': ' '.join(list(words)), 'start': 0, 'end': 0, 'index': i} for i, words
                                   in enumerate(get_slices(origin_text.split(), 2))]

            json.dump(origin_text_phrases, open(os.path.join(data_dir, 'origin_text_phrases.json'), 'w'), indent=4,
                      ensure_ascii=False)

            build_lm(args, data_lower, vocab_str)

        if 3 in _steps:
            # create fst
            out_file = os.path.join(data_dir, 'words_trie.fst')
            fst_vocab = os.path.join(data_dir, f'vocab-{args.top_k}.txt')
            token_file = 'labels.txt'
            build_fst_from_words_file(fst_vocab, token_file, out_file)

        if 4 in _steps:
            # decode
            frame_segments = json.load(open(os.path.join(data_dir, 'frame_segments.json'), 'r'))
            char_ms, transcript = real_ctc_decode(filename, data_dir, frame_segments, model_name, mode=ctc_mode)
            Path(data_dir, 'results', model_name, ctc_mode).mkdir(parents=True, exist_ok=True)
            np.savetxt(os.path.join(data_dir, 'results', model_name, ctc_mode, 'char_ms.txt'), char_ms)
            with open(os.path.join(data_dir, 'results', model_name, ctc_mode, 'transcript.txt'), 'w') as f:
                f.write(transcript)


if __name__ == '__main__':
    main()
