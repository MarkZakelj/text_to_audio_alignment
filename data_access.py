import os
from pathlib import Path
import json
import configparser


def get_config():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config


def list_sample_names():
    sample_names = [name.name for name in Path('audio_samples/').iterdir()]
    return sample_names


def list_model_names():
    model_names = [name.stem for name in Path('nemo_models').iterdir()]
    return model_names


def get_reference(sample_name):
    reference = json.load(open(os.path.join('audio_samples', sample_name, 'aligned_reference.json')))
    return reference


def get_alignment(sample_name, model_name, ctc_mode):
    alignment = json.load(
        open(os.path.join('audio_samples', sample_name, 'results', model_name, ctc_mode, 'aligned_words.json')))
    return alignment


def save_eval_result(result, sample_name, model_name, ctc_mode):
    json.dump(result,
              open(os.path.join('audio_samples', sample_name, 'results', model_name, ctc_mode, 'eval_result.json'),
                   'w'),
              ensure_ascii=False, indent=4)


def get_eval_results(sample_name, model_name, ctc_mode):
    results = json.load(
        open(os.path.join('audio_samples', sample_name, 'results', model_name, ctc_mode, 'eval_result.json')))
    return results


if __name__ == '__main__':
    print(list_model_names())
