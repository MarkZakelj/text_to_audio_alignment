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
import data_access as da
from generate_lm import build_lm, convert_and_filter_topk
import argparse
import re
from tqdm import tqdm
import multiprocessing as mp


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


def to_dashed_text(text, group_size):
    dashed_text = ''
    for _slice in get_slices(re.findall(r"[\w']+", text), group_size):
        dashed_text += '-'.join(_slice) + ' '
    return dashed_text.rstrip() + '\n'


def min_segment(segments, size_function):
    min_idx = 0
    min_size = size_function(segments[0])
    for i, seg in enumerate(segments):
        size = size_function(seg)
        if size < min_size:
            min_idx = i
            min_size = size
    return segments[min_idx], min_idx


def merge_segments(segments, min_len=10):
    left, right = -1, 1

    def size(x):
        return x[1] - x[0]

    tmp_segments = [e for e in segments]
    min_seg, min_idx = min_segment(tmp_segments, size_function=size)
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
        min_seg, min_idx = min_segment(tmp_segments, size_function=size)
    return tmp_segments


def merge_vocab_str(vocab_str, min_len=4):
    left, right = -1, 1
    tmp_segments = [e for e in vocab_str]
    min_seg, min_idx = min_segment(tmp_segments, len)
    while len(min_seg) < min_len and len(tmp_segments) > 1:
        if min_idx == 0:
            direction = right
        elif min_idx == len(tmp_segments) - 1:
            direction = left
        else:
            left_size = len(tmp_segments[min_idx - 1])
            right_size = len(tmp_segments[min_idx + 1])
            if left_size < right_size:
                direction = left
            else:
                direction = right
        first, second = min_idx, min_idx + direction  # if direction == right
        if direction == left:
            first, second = second, first
        tmp_segments[first] = tmp_segments[first] + ' ' + tmp_segments[second]
        del tmp_segments[second]
        min_seg, min_idx = min_segment(tmp_segments, len)
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


def decode_segment(inputs, endpoints, mode, tokenizer, scorer, beam_size):
    start, end = endpoints
    if mode.startswith('lm'):
        result = decoders.ctc_beam_search_decoder(inputs[start:end, :],
                                                  lm_scorer=scorer,
                                                  blank=tokenizer.blank_idx,
                                                  beam_size=beam_size)
        best_result = result[0]
    else:
        raise ValueError('mode must be one of: "greedy" | "lm+fst" |')

    return best_result


def real_ctc_decode(filename, data_dir, merged_segments, model_name, mode='lm+fst', beam_size=100):
    data = np.load(os.path.join(data_dir, 'results', model_name, 'output.npy'))

    inputs = torch.as_tensor(data).log_softmax(1)

    tokenizer = CharTokenizer('labels.txt')

    # greedy using beam
    greedy_result = decoders.ctc_greedy_decoder(inputs[merged_segments[0][0]:merged_segments[-1][1], :],
                                                blank=tokenizer.blank_idx)
    txt_result_greedy = ''.join(tokenizer.idx2token(greedy_result[1]))

    print(txt_result_greedy)
    transcription_json = []




    transcription_json = []
    res = []
    frame_idss = []
    if mode == 'lm':
        scorer = KenLMScorer(os.path.join(data_dir, 'lm.arpa'),
                             tokenizer,
                             alpha=1.9,
                             beta=0.3,
                             unit='word')
        with mp.Pool() as p:
            best_results = p.starmap(decode_segment,
                                     [(inputs, endpoints, mode, tokenizer, scorer, beam_size) for endpoints in
                                      merged_segments])
    elif mode.startswith('lm+fst'):
        scorer = KenLMScorer(os.path.join(data_dir, 'lm.arpa'),
                             tokenizer,
                             trie_path=os.path.join(data_dir, 'words_trie.fst'),
                             alpha=0.01,
                             beta=0.0,
                             unit='word')
        with mp.Pool(10) as p:
            best_results = p.starmap(decode_segment,
                                     [(inputs, endpoints, mode, tokenizer, scorer, beam_size) for endpoints in
                                      merged_segments])
    elif mode == 'greedy':
        best_results = [decoders.ctc_greedy_decoder(inputs[start:stop, :], blank=tokenizer.blank_idx) for start, stop in
                        merged_segments]
    else:
        raise ValueError('ctc_mode is invalid')
    for best_result, endpoints in zip(best_results, merged_segments):
        start, stop = endpoints
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
        print(txt_result)

        word_starts = [0] + list(frame_ids[np.where(best_token_ids == 0)[0] + 1] * 0.02)
        word_ends = list(frame_ids[np.where(best_token_ids == 0)[0] - 1] * 0.02) + [frame_ids[-1] * 0.02]
        transcription_json += [{'transcript': triple[0], 'start': triple[1], 'end': triple[2]}
                               for triple in zip(txt_result.split(' '), word_starts, word_ends)]
    return np.array(frame_idss) * 0.02, ' '.join(res)


def main():
    config = da.get_config()
    sample_name = config.get('main', 'sample_name')
    model = config.get('main', 'model_name').replace('.nemo', '')
    mode = config.get('main', 'ctc_mode')
    samples = [sample_name] if sample_name != 'all' else da.list_sample_names()
    models = [model] if model != 'all' else da.list_model_names()
    ctc_modes = [mode] if mode != 'all' else ['greedy', 'lm+fst']
    dashes_group_size = config.getint('decode', 'dashes_group_size')
    beam_size = config.getint('decode', 'beam_size')
    min_segment_len = config.getfloat('decode', 'min_segment_len')

    _steps = {2, 3, 4}
    for sample in samples:
        data_dir = os.path.join('audio_samples', sample)
        frame_segments = json.load(open(os.path.join(data_dir, 'frame_segments.json'), 'r'))
        merged_segments = merge_segments(frame_segments, min_segment_len)
        for model_name in models:
            for ctc_mode in ctc_modes:
                if 2 in _steps:
                    # generate lm
                    args = argparse.Namespace()
                    args.input_txt = os.path.join(data_dir, 'main_text.txt')
                    args.output_dir = data_dir
                    args.top_k = 5000
                    args.kenlm_bins = os.path.join('kenlm', 'build', 'bin')
                    args.arpa_order = 2
                    args.max_arpa_memory = "80%"
                    args.binary_a_bits = 255
                    args.binary_q_bits = 8
                    args.binary_type = 'trie'
                    args.discount_fallback = True

                    data_lower, vocab_str = convert_and_filter_topk(args)

                    dashed_text = to_dashed_text(da.get_data_lower(sample), dashes_group_size)
                    print(dashed_text)
                    with gzip.open(os.path.join(data_dir, 'lower_dashes.txt.gz'), 'w') as f:
                        f.write(dashed_text.replace('-', ' ').encode('utf-8'))
                    vocab_str_lm = '\n'.join(dashed_text.rstrip().split(' ')).replace('-', ' ') + '\n'
                    if ctc_mode == 'lm+fst+dashes':
                        vocab_as_segments = []
                        clean_text = da.get_data_lower(sample).strip().replace('\n', ' ').replace('  ', ' ').split(' ')
                        vocab_str_merged = '\n'.join(merge_vocab_str(clean_text))
                        print(vocab_str_merged)
                        # for w_idx in range(0, len(clean_text), dashes_group_size):
                        #     if w_idx + dashes_group_size <= len(clean_text):
                        #         vocab_str_fst.append(' '.join(clean_text[w_idx:w_idx + dashes_group_size]))
                        #     else:
                        #         vocab_str_fst.append(' '.join(clean_text[w_idx:]))
                        # print(vocab_str_fst)
                        # vocab_str_fst = '\n'.join(vocab_str_fst)
                        with open(os.path.join(data_dir, 'vocab_dashes_fst.txt'), 'w') as f:
                            f.write(vocab_str_merged)
                        # data_lower = os.path.join(data_dir, 'lower_dashes.txt.gz')
                        # print(data_lower, vocab_str_fst)
                        build_lm(args, data_lower, vocab_str_merged)
                    else:
                        build_lm(args, data_lower, vocab_str)

                if 3 in _steps:
                    # create fst
                    out_file = os.path.join(data_dir, 'words_trie.fst')
                    vocab_filename = f'vocab-{args.top_k}.txt'
                    if ctc_mode == 'lm+fst+dashes':
                        vocab_filename = 'vocab_dashes_fst.txt'
                    fst_vocab = os.path.join(data_dir, vocab_filename)
                    token_file = 'labels.txt'
                    build_fst_from_words_file(fst_vocab, token_file, out_file)

                if 4 in _steps:
                    # decode

                    char_ms, transcript = real_ctc_decode(sample, data_dir, merged_segments, model_name, mode=ctc_mode,
                                                          beam_size=beam_size)
                    print(transcript)
                    Path(data_dir, 'results', model_name, ctc_mode).mkdir(parents=True, exist_ok=True)
                    np.savetxt(os.path.join(data_dir, 'results', model_name, ctc_mode, 'char_ms.txt'), char_ms)
                    with open(os.path.join(data_dir, 'results', model_name, ctc_mode, 'transcript.txt'), 'w') as f:
                        f.write(transcript)


if __name__ == '__main__':
    main()
