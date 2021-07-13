import torch
from alignment.audio import DEFAULT_RATE, read_frames_from_file, vad_split
from nemo_infer import infer
import numpy as np
from pathlib import Path
import json
import argparse
import sox
from nemo.collections.asr.models import EncDecCTCModel

stt_min_duration_ms = 300
stt_max_duration_ms = None
model_format = (DEFAULT_RATE, 1, 2)


# model_path = 'nemo_models/trained_33307b3_conf-sm.nemo'


def segment_and_asr(audio_path, model):
    """
    Reads the audio file, calculates log-probabilities of tokens using a trained nemo model and segments the audio file into smaller sections

    :returns
    log_probabilities of tokens in individual frames
    step size of frames (in seconds)
    vocabulary (list of tokens)
    list of segments in the audio file (start, end in seconds)
    """

    sr = sox.file_info.sample_rate(audio_path)
    nsamples = np.round(sox.file_info.duration(audio_path) * sr)

    if isinstance(model, str):
        asr_model = EncDecCTCModel.restore_from(model)
    else:
        asr_model = model
    _, log_prob, vocab = infer(asr_model, [audio_path], 1)

    step = np.round(nsamples / len(log_prob[0]))

    log_prob_step_sec = step / sr
    # Run VAD on the input file
    frames = read_frames_from_file(audio_path, model_format, frame_duration_ms=20)
    split = vad_split(frames, model_format, threshold=0.5, aggressiveness=2)
    segments = []
    for i, segment in enumerate(split):
        segment_buffer, time_start, time_end = segment
        time_length = time_end - time_start
        if stt_min_duration_ms and time_length < stt_min_duration_ms:
            print('Fragment {}: Audio too short for STT'.format(i))
            continue
        if stt_max_duration_ms and time_length > stt_max_duration_ms:
            print('Fragment {}: Audio too long for STT'.format(i))
            continue
        segments.append((time_start, time_end))

    return log_prob[0], log_prob_step_sec, vocab, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Sample name. If "all", select every possible sample',
                        required=True)
    parser.add_argument('--model', type=str, help='Name of nemo model without extension, e.g "trained_model"',
                        required=True)
    args = parser.parse_args()
    sample_name = args.name
    model_name = args.model.replace('.nemo', '')
    model_name_w_ext = model_name + '.nemo'
    model_path = Path('nemo_models', model_name_w_ext)
    asr_model = EncDecCTCModel.restore_from(str(model_path))
    if not model_path.exists():
        possible_models = [m.stem for m in Path('nemo_models').iterdir()]
        raise FileNotFoundError(f'Model "{model_path.stem}" does not exist. Possible models: {possible_models}')
    if sample_name != 'all':
        filenames = [sample_name]
    else:
        filenames = [name.name for name in Path('audio_samples/').iterdir()]
    for filename in filenames:
        print('\n' + filename)
        torch.cuda.empty_cache()
        data_dir = Path('audio_samples', filename)
        audio_filename = Path(data_dir, 'main_audio.wav')

        # get model output
        log_prob, log_prob_step_sec, vocab, segments = segment_and_asr(str(audio_filename), asr_model)

        # convert segment endpoints to frame index
        print(segments, log_prob_step_sec)
        _k = log_prob_step_sec * (10 ** 3)
        frame_segments = [[int(e[0] / _k), int(e[1] / _k)] for e in segments]
        print(frame_segments)

        # save frame index to json file
        json.dump(frame_segments, open(Path(data_dir, 'frame_segments.json'), 'w'), indent=4)

        # create directory for model output (if it doesn't exist)
        model_output_path = Path(data_dir, 'results', model_name)
        Path.mkdir(model_output_path, parents=True, exist_ok=True)

        # save model output as npy file
        np.save(Path(model_output_path, 'output.npy').as_posix(), log_prob)


if __name__ == '__main__':
    main()
