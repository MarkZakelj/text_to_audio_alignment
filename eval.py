import argparse
import json
import numpy as np
import os
from pathlib import Path
import data_access as da


def mae(alignment, reference):
    errors = []
    errors_start = []
    bias_start = 0
    errors_end = []
    bias_end = 0
    if len(alignment) != len(reference):
        print(f'Different length: reference: {len(reference)}, alignment: {len(alignment)}')
    for i, (al, ref) in enumerate(zip(alignment, reference)):
        mae_start = abs(ref['start'] - al['start'])
        mae_end = abs(ref['end'] - al['end'])
        errors_start.append(mae_start)
        errors_end.append(mae_end)
        bias_start += (al['start'] - ref['start'])
        bias_end += (al['end'] - ref['end'])
        if ref['text'] != al['text']:
            print(f'{ref["text"]} != {al["text"]}, {i}')
        else:
            errors.append({'text': ref['text'], 'ae_start': mae_start, 'ae_end': mae_end})
    bias_start /= len(alignment)
    bias_end /= len(alignment)
    return errors, bias_start, bias_end


def main():
    config = da.get_config()
    sample_name = config.get('main', 'sample_name')
    model = config.get('main', 'model_name').replace('.nemo', '')
    mode = config.get('main', 'ctc_mode')
    samples = [sample_name] if sample_name != 'all' else da.list_sample_names()
    models = [model] if model != 'all' else da.list_model_names()
    ctc_modes = [mode] if mode != 'all' else ['greedy', 'lm+fst']
    for sample in samples:
        for model in models:
            for ctc_mode in ctc_modes:
                calculated = da.get_alignment(sample, model, ctc_mode)
                reference = da.get_reference(sample)
                print(f"\nCalculating MAE for {sample}, {model}, {ctc_mode}")
                result, bias_s, bias_e = mae(calculated, reference)
                da.save_eval_result(result, sample, model, ctc_mode)
                all_positions = np.array([e[position] for e in result for position in ['ae_start', 'ae_end']])
                print('MAE: {}, STD: {}'.format(all_positions.mean(), all_positions.std()))
                print(f'bias_start: {bias_s}, bias_end: {bias_e}')
                print('top 20 wrong words')
                worst_results = sorted([(e['text'], (e['ae_start'] + e['ae_end']) / 2) for e in result],
                                       key=lambda x: x[1], reverse=True)
                print(worst_results[:20])


if __name__ == '__main__':
    main()
