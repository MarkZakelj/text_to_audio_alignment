import argparse
import json
import os
import re
import numpy as np
from pathlib import Path
from shutil import copyfile

from alignment.alignment import align, NAMED_NUMBERS
from alignment.text import Alphabet


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description='Force align speech data with a transcript.')

    parser.add_argument('--force', action="store_true",
                        help='Overwrite existing files')
    parser.add_argument('--loglevel', type=int, required=False, default=20,
                        help='Log level (between 0 and 50) - default: 20')
    parser.add_argument('--no-progress', action="store_true",
                        help='Prevents showing progress indication')
    parser.add_argument('--progress-interval', type=float, default=1.0,
                        help='Progress indication interval in seconds')
    parser.add_argument('--text-context', type=int, required=False, default=10,
                        help='Size of textual context for logged statements - default: 10')
    parser.add_argument('--alphabet', required=False,
                        help='Path to an alphabet file (overriding the one from --stt-model-dir)')

    text_group = parser.add_argument_group(title='Text pre-processing options')
    text_group.add_argument('--text-meaningful-newlines', action="store_true",
                            help='Newlines from plain text file separate phrases/speakers. '
                                 '(see --align-phrase-snap-factor)')
    text_group.add_argument('--text-keep-dashes', action="store_true",
                            help='No replacing of dashes with spaces. Dependent of alphabet if kept at all.')
    text_group.add_argument('--text-keep-ws', action="store_true",
                            help='No normalization of whitespace. Keep it as it is.')
    text_group.add_argument('--text-keep-casing', action="store_true",
                            help='No lower-casing of characters. Keep them as they are.')

    align_group = parser.add_argument_group(title='Alignment algorithm options')
    align_group.add_argument('--align-workers', type=int, required=False,
                             help='Number of parallel alignment workers - defaults to number of CPUs')
    align_group.add_argument('--align-max-candidates', type=int, required=False, default=10,
                             help='How many global 3gram match candidates are tested at max (default: 10)')
    align_group.add_argument('--align-candidate-threshold', type=float, required=False, default=0.92,
                             help='Factor for how many 3grams the next candidate should have at least ' +
                                  'compared to its predecessor (default: 0.92)')
    align_group.add_argument('--align-match-score', type=int, required=False, default=100,
                             help='Matching score for Smith-Waterman alignment (default: 100)')
    align_group.add_argument('--align-mismatch-score', type=int, required=False, default=-100,
                             help='Mismatch score for Smith-Waterman alignment (default: -100)')
    align_group.add_argument('--align-gap-score', type=int, required=False, default=-100,
                             help='Gap score for Smith-Waterman alignment (default: -100)')
    align_group.add_argument('--align-shrink-fraction', type=float, required=False, default=0.1,
                             help='Length fraction of the fragment that it could get shrinked during fine alignment')
    align_group.add_argument('--align-stretch-fraction', type=float, required=False, default=0.25,
                             help='Length fraction of the fragment that it could get stretched during fine alignment')
    align_group.add_argument('--align-word-snap-factor', type=float, required=False, default=1.5,
                             help='Priority factor for snapping matched texts to word boundaries '
                                  '(default: 1.5 - slightly snappy)')
    align_group.add_argument('--align-phrase-snap-factor', type=float, required=False, default=1.0,
                             help='Priority factor for snapping matched texts to word boundaries '
                                  '(default: 1.0 - no snapping)')
    align_group.add_argument('--align-similarity-algo', type=str, required=False, default='wng',
                             help='Similarity algorithm during fine-alignment - one of '
                                  'wng|editex|levenshtein|mra|hamming|jaro_winkler (default: wng)')
    align_group.add_argument('--align-wng-min-size', type=int, required=False, default=1,
                             help='Minimum N-gram size for weighted N-gram similarity '
                                  'during fine-alignment (default: 1)')
    align_group.add_argument('--align-wng-max-size', type=int, required=False, default=3,
                             help='Maximum N-gram size for weighted N-gram similarity '
                                  'during fine-alignment (default: 3)')
    align_group.add_argument('--align-wng-size-factor', type=float, required=False, default=1,
                             help='Size weight for weighted N-gram similarity '
                                  'during fine-alignment (default: 1)')
    align_group.add_argument('--align-wng-position-factor', type=float, required=False, default=2.5,
                             help='Position weight for weighted N-gram similarity '
                                  'during fine-alignment (default: 2.5)')

    output_group = parser.add_argument_group(title='Output options')
    output_group.add_argument('--output-pretty', action="store_true",
                              help='Writes indented JSON output"')

    for short in NAMED_NUMBERS.keys():
        long, atype, desc = NAMED_NUMBERS[short]
        desc = (' - value range: ' + desc) if desc else ''
        output_group.add_argument('--output-' + short.lower(), action="store_true",
                                  help='Writes {} ({}) to output'.format(long, short))
        for extreme in ['Min', 'Max']:
            output_group.add_argument('--output-' + extreme.lower() + '-' + short.lower(), type=atype, required=False,
                                      help='{}imum {} ({}) the STT transcript of the audio '
                                           'has to have when compared with the original text{}'
                                      .format(extreme, long, short, desc))

    return parser.parse_args(args=args, namespace=namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Primer, ki se bo procesiral. Ce je all, izberi vse',
                        required=True)
    parser.add_argument('-m', '--mode', type=str, help='nacin izvajanja. "greedy" ali "lm+fst"', required=True)
    parser.add_argument('--model', type=str, help='Name of asr model used', required=True)
    args = parser.parse_args()
    sample_name = args.name
    ctc_mode = args.mode
    model_name = args.model.replace('.nemo', '')
    if sample_name != 'all':
        filenames = [sample_name]
    else:
        filenames = [name.name for name in Path('audio_samples/').iterdir()]
    alphabet = Alphabet('alphabet.txt')
    for filename in filenames:
        data_dir = os.path.join('audio_samples', filename)
        char_ms = np.loadtxt(os.path.join(data_dir, 'results', model_name, ctc_mode, 'char_ms.txt'))
        copyfile(f'{data_dir}/origin_text_phrases.json', f'{data_dir}/origin_fragments.json')
        dropped_fragments = 1
        align_ns = argparse.Namespace()
        align_ns.script = os.path.join(data_dir, 'results', model_name, ctc_mode, 'transcript.txt')
        align_ns.tlog = os.path.join(data_dir, 'origin_fragments.json')
        align_ns.aligned = os.path.join(data_dir, 'results', model_name, ctc_mode, 'aligned_phrases.json')
        align_ns.force = True
        align_ns.output_pretty = True
        args = parse_args(args=[], namespace=align_ns)
        while dropped_fragments > 0:
            align(alphabet, args)
            origin_fragments = json.load(open(f'{data_dir}/origin_fragments.json', 'r'))
            n_fragments = len(origin_fragments)
            aligned_fragments = json.load(open(align_ns.aligned, 'r'))
            matched_ids = set([e['index'] for e in aligned_fragments])
            non_matched_ids = set([idx for idx in range(n_fragments) if idx not in matched_ids])
            dropped_fragments = n_fragments - len(aligned_fragments)
            if dropped_fragments > 0:
                merges = []
                for i in range(n_fragments):
                    if i not in non_matched_ids:
                        continue
                    if i == n_fragments - 1:
                        target = i - 1
                    else:
                        target = i + 1
                    merges.append((min(i, target), max(i, target)))
                    non_matched_ids.discard(i)
                    non_matched_ids.discard(target)
                origin_unmerged = dict([(e['index'], e['transcript']) for e in origin_fragments])
                for i, i_next in merges:
                    text_i_next = origin_unmerged[i_next]
                    text_i = origin_unmerged.pop(i)
                    origin_unmerged[i_next] = text_i.rstrip() + ' ' + text_i_next.lstrip()
                origin_merged = [{'transcript': val, 'start': 0, 'end': 0, 'index': i} for i, (k, val) in
                                 enumerate(origin_unmerged.items())]
                json.dump(origin_merged, open(f'{data_dir}/origin_fragments.json', 'w'), ensure_ascii=False,
                          indent=4)

        aligned = json.load(open(os.path.join(data_dir, 'results', model_name, ctc_mode, 'aligned_phrases.json'), 'r'))
        alinged_true = []
        word_index = 0
        for el in aligned:
            words = el['transcript']
            l0 = len(words)
            l1 = len(el['aligned'])
            start = el['text-start']
            starts = [start]
            ends = []
            for x in re.finditer(' ', words):
                _start = start + int(x.span()[0] / l0 * l1)
                _end = start + int((x.span()[0] - 1) / l0 * l1)
                starts.append(_start)
                ends.append(_end)
            ends.append(start + l1 - 1)
            for i, word in enumerate(words.split()):
                ms_start = char_ms[starts[i] if starts[i] < len(char_ms) else -1]
                ms_end = char_ms[ends[i] if ends[i] < len(char_ms) else -1]
                if ms_start == ms_end:
                    ms_end += 0.01
                alinged_true.append(
                    {'text': word, 'start': float(ms_start), 'end': float(ms_end), 'index': word_index})
                word_index += 1
        json.dump(alinged_true,
                  open(os.path.join(data_dir, 'results', model_name, ctc_mode, 'aligned_words.json'), 'w'), indent=4,
                  ensure_ascii=False)


if __name__ == '__main__':
    main()
