import json
import logging
from .search import FuzzySearch
from .text import TextCleaner, levenshtein, similarity
from .utils import enweight
import textdistance
from collections import Counter

ALGORITHMS = ['WNG', 'jaro_winkler', 'editex', 'levenshtein', 'mra', 'hamming']
SIM_DESC = 'From 0.0 (not equal at all) to 100.0 (totally equal)'
NAMED_NUMBERS = {
    'tlen': ('transcript length', int, None),
    'mlen': ('match length', int, None),
    'SWS': ('Smith-Waterman score', float, 'From 0.0 (not equal at all) to 100.0+ (pretty equal)'),
    'WNG': ('weighted N-gram similarity', float, SIM_DESC),
    'jaro_winkler': ('Jaro-Winkler similarity', float, SIM_DESC),
    'editex': ('Editex similarity', float, SIM_DESC),
    'levenshtein': ('Levenshtein similarity', float, SIM_DESC),
    'mra': ('MRA similarity', float, SIM_DESC),
    'hamming': ('Hamming similarity', float, SIM_DESC),
    'CER': ('character error rate', float, 'From 0.0 (no different words) to 100.0+ (total miss)'),
    'WER': ('word error rate', float, 'From 0.0 (no wrong characters) to 100.0+ (total miss)')
}


def read_script(script_path, alphabet, args):
    tc = TextCleaner(alphabet,
                     dashes_to_ws=not args.text_keep_dashes,
                     normalize_space=not args.text_keep_ws,
                     to_lower=not args.text_keep_casing)
    with open(script_path, 'r', encoding='utf-8') as script_file:
        content = script_file.read()
        if script_path.endswith('.script'):
            for phrase in json.loads(content):
                tc.add_original_text(phrase['text'], meta=phrase)
        elif args.text_meaningful_newlines:
            for phrase in content.split('\n'):
                tc.add_original_text(phrase)
        else:
            tc.add_original_text(content)
    return tc


def align(alphabet, args):
    tlog, script, aligned = args.tlog, args.script, args.aligned

    logging.debug("Loading script from %s..." % script)
    tc = read_script(script, alphabet, args)
    search = FuzzySearch(tc.clean_text,
                         max_candidates=args.align_max_candidates,
                         candidate_threshold=args.align_candidate_threshold,
                         match_score=args.align_match_score,
                         mismatch_score=args.align_mismatch_score,
                         gap_score=args.align_gap_score)

    logging.debug("Loading transcription log from %s..." % tlog)
    with open(tlog, 'r', encoding='utf-8') as transcription_log_file:
        fragments = json.load(transcription_log_file)
    for index, fragment in enumerate(fragments):
        meta = {}
        for key, value in list(fragment.items()):
            if key not in ['start', 'end', 'transcript']:
                meta[key] = value
                del fragment[key]
        fragment['meta'] = meta
        fragment['index'] = index
        fragment['transcript'] = fragment['transcript'].strip()

    reasons = Counter()

    def skip(index, reason):
        logging.info('Fragment {}: {}'.format(index, reason))
        reasons[reason] += 1

    def split_match(fragments, start=0, end=-1):
        n = len(fragments)
        if n < 1:
            return
        elif n == 1:
            weighted_fragments = [(0, fragments[0])]
        else:
            # so we later know the original index of each fragment
            weighted_fragments = enumerate(fragments)
            # assigns high values to long statements near the center of the list
            weighted_fragments = enweight(weighted_fragments)
            weighted_fragments = map(lambda fw: (fw[0], (1 - fw[1]) * len(fw[0][1]['transcript'])), weighted_fragments)
            # fragments with highest weights first
            weighted_fragments = sorted(weighted_fragments, key=lambda fw: fw[1], reverse=True)
            # strip weights
            weighted_fragments = list(map(lambda fw: fw[0], weighted_fragments))
        for index, fragment in weighted_fragments:
            match = search.find_best(fragment['transcript'], start=start, end=end)
            match_start, match_end, sws_score, match_substitutions = match
            if sws_score > (n - 1) / (2 * n):
                fragment['match-start'] = match_start
                fragment['match-end'] = match_end
                fragment['sws'] = sws_score
                fragment['substitutions'] = match_substitutions
                for f in split_match(fragments[0:index], start=start, end=match_start):
                    yield f
                yield fragment
                for f in split_match(fragments[index + 1:], start=match_end, end=end):
                    yield f
                return
        for _, _ in weighted_fragments:
            yield None

    matched_fragments = split_match(fragments)
    matched_fragments = list(filter(lambda f: f is not None, matched_fragments))

    similarity_algos = {}

    def phrase_similarity(algo, a, b):
        if algo in similarity_algos:
            return similarity_algos[algo](a, b)
        algo_impl = lambda aa, bb: None
        if algo.lower() == 'wng':
            algo_impl = similarity_algos[algo] = lambda aa, bb: similarity(
                aa,
                bb,
                direction=1,
                min_ngram_size=args.align_wng_min_size,
                max_ngram_size=args.align_wng_max_size,
                size_factor=args.align_wng_size_factor,
                position_factor=args.align_wng_position_factor)
        elif algo in ALGORITHMS:
            algo_impl = similarity_algos[algo] = getattr(textdistance, algo).normalized_similarity
        else:
            logging.fatal('Unknown similarity metric "{}"'.format(algo))
            exit(1)
        return algo_impl(a, b)

    def get_similarities(a, b, n, gap_text, gap_meta, direction):
        if direction < 0:
            a, b, gap_text, gap_meta = a[::-1], b[::-1], gap_text[::-1], gap_meta[::-1]
        similarities = list(map(
            lambda i: (args.align_word_snap_factor if gap_text[i + 1] == ' ' else 1) *
                      (args.align_phrase_snap_factor if gap_meta[i + 1] is None else 1) *
                      (phrase_similarity(args.align_similarity_algo, a, b + gap_text[1:i + 1])),
            range(n)))
        best = max((v, i) for i, v in enumerate(similarities))[1] if n > 0 else 0
        return best, similarities

    for index in range(len(matched_fragments) + 1):
        if index > 0:
            a = matched_fragments[index - 1]
            a_start, a_end = a['match-start'], a['match-end']
            a_len = a_end - a_start
            a_stretch = int(a_len * args.align_stretch_fraction)
            a_shrink = int(a_len * args.align_shrink_fraction)
            a_end = a_end - a_shrink
            a_ext = a_shrink + a_stretch
        else:
            a = None
            a_start = a_end = 0
        if index < len(matched_fragments):
            b = matched_fragments[index]
            b_start, b_end = b['match-start'], b['match-end']
            b_len = b_end - b_start
            b_stretch = int(b_len * args.align_stretch_fraction)
            b_shrink = int(b_len * args.align_shrink_fraction)
            b_start = b_start + b_shrink
            b_ext = b_shrink + b_stretch
        else:
            b = None
            b_start = b_end = len(search.text)

        assert a_end <= b_start
        assert a_start <= a_end
        assert b_start <= b_end
        if a_end == b_start or a_start == a_end or b_start == b_end:
            continue
        gap_text = tc.clean_text[a_end - 1:b_start + 1]
        gap_meta = tc.meta[a_end - 1:b_start + 1]

        if a:
            a_best_index, a_similarities = get_similarities(a['transcript'],
                                                            tc.clean_text[a_start:a_end],
                                                            min(len(gap_text) - 1, a_ext),
                                                            gap_text,
                                                            gap_meta,
                                                            1)
            a_best_end = a_best_index + a_end
        if b:
            b_best_index, b_similarities = get_similarities(b['transcript'],
                                                            tc.clean_text[b_start:b_end],
                                                            min(len(gap_text) - 1, b_ext),
                                                            gap_text,
                                                            gap_meta,
                                                            -1)
            b_best_start = b_start - b_best_index

        if a and b and a_best_end > b_best_start:
            overlap_start = b_start - len(b_similarities)
            a_similarities = a_similarities[overlap_start - a_end:]
            b_similarities = b_similarities[:len(a_similarities)]
            best_index = max((sum(v), i) for i, v in enumerate(zip(a_similarities, b_similarities)))[1]
            a_best_end = b_best_start = overlap_start + best_index

        if a:
            a['match-end'] = a_best_end
        if b:
            b['match-start'] = b_best_start

    def apply_number(number_key, index, fragment, show, get_value):
        kl = number_key.lower()
        should_output = getattr(args, 'output_' + kl)
        min_val, max_val = getattr(args, 'output_min_' + kl), getattr(args, 'output_max_' + kl)
        if kl.endswith('len') and min_val is None:
            min_val = 1
        if should_output or min_val or max_val:
            val = get_value()
            if not kl.endswith('len'):
                show.insert(0, '{}: {:.2f}'.format(number_key, val))
                if should_output:
                    fragment[kl] = val
            reason_base = '{} ({})'.format(NAMED_NUMBERS[number_key][0], number_key)
            reason = None
            if min_val and val < min_val:
                reason = reason_base + ' too low'
            elif max_val and val > max_val:
                reason = reason_base + ' too high'
            if reason:
                skip(index, reason)
                return True
        return False

    substitutions = Counter()
    result_fragments = []
    for fragment in matched_fragments:
        index = fragment['index']
        time_start = fragment['start']
        time_end = fragment['end']
        fragment_transcript = fragment['transcript']
        result_fragment = {
            'start': time_start,
            'end': time_end,
            'index': index,
        }
        sample_numbers = []

        if apply_number('tlen', index, result_fragment, sample_numbers, lambda: len(fragment_transcript)):
            continue
        result_fragment['transcript'] = fragment_transcript

        if 'match-start' not in fragment or 'match-end' not in fragment:
            skip(index, 'No match for transcript')
            continue
        match_start, match_end = fragment['match-start'], fragment['match-end']
        if match_end - match_start <= 0:
            skip(index, 'Empty match for transcript')
            continue
        original_start = tc.get_original_offset(match_start)
        original_end = tc.get_original_offset(match_end)
        result_fragment['text-start'] = original_start
        result_fragment['text-end'] = original_end

        meta_dict = {}
        for meta in list(tc.collect_meta(match_start, match_end)) + [fragment['meta']]:
            for key, value in meta.items():
                if key == 'text':
                    continue
                if key in meta_dict:
                    values = meta_dict[key]
                else:
                    values = meta_dict[key] = []
                if value not in values:
                    values.append(value)
        result_fragment['meta'] = meta_dict

        result_fragment['aligned-raw'] = tc.original_text[original_start:original_end]

        fragment_matched = tc.clean_text[match_start:match_end]
        if apply_number('mlen', index, result_fragment, sample_numbers, lambda: len(fragment_matched)):
            continue
        result_fragment['aligned'] = fragment_matched

        if apply_number('SWS', index, result_fragment, sample_numbers, lambda: 100 * fragment['sws']):
            continue

        should_skip = False
        for algo in ALGORITHMS:
            should_skip = should_skip or apply_number(algo, index, result_fragment, sample_numbers,
                                                      lambda: 100 * phrase_similarity(algo,
                                                                                      fragment_matched,
                                                                                      fragment_transcript))
        if should_skip:
            continue

        if apply_number('CER', index, result_fragment, sample_numbers,
                        lambda: 100 * levenshtein(fragment_transcript, fragment_matched) /
                                len(fragment_matched)):
            continue

        if apply_number('WER', index, result_fragment, sample_numbers,
                        lambda: 100 * levenshtein(fragment_transcript.split(), fragment_matched.split()) /
                                len(fragment_matched.split())):
            continue

        substitutions += fragment['substitutions']

        result_fragments.append(result_fragment)
        logging.debug('Fragment %d aligned with %s' % (index, ' '.join(sample_numbers)))
        logging.debug('- T: ' + args.text_context * ' ' + '"%s"' % fragment_transcript)
        logging.debug('- O: %s|%s|%s' % (
            tc.clean_text[match_start - args.text_context:match_start],
            fragment_matched,
            tc.clean_text[match_end:match_end + args.text_context]))
    with open(aligned, 'w', encoding='utf-8') as result_file:
        result_file.write(json.dumps(result_fragments, indent=4 if args.output_pretty else None, ensure_ascii=False))
    return aligned, len(result_fragments), len(fragments) - len(result_fragments), reasons
