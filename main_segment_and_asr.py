import torch
from alignment.audio import DEFAULT_RATE, read_frames_from_file, vad_split
from nemo_infer import infer
import numpy as np
from pathlib import Path
import json
import data_access as da
import sox
from nemo.collections.asr.models import EncDecCTCModel

stt_min_duration_ms = 300
stt_max_duration_ms = None
model_format = (DEFAULT_RATE, 1, 2)


# model_path = 'nemo_models/trained_33307b3_conf-sm.nemo'


def segment_and_asr(audio_path, model, threshold, aggressiveness):
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
    split = vad_split(frames, model_format, threshold=threshold, aggressiveness=aggressiveness)
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
    config = da.get_config()
    sample_name = config.get('main', 'sample_name')
    model = config.get('main', 'model_name').replace('.nemo', '')
    threshold = config.getfloat('segment', 'threshold')
    aggressiveness = config.getint('segment', 'aggressiveness')
    samples = [sample_name] if sample_name != 'all' else da.list_sample_names()
    models = [model] if model != 'all' else da.list_model_names()
    for model_name in models:
        model_name_w_ext = model_name + '.nemo'
        model_path = Path('nemo_models', model_name_w_ext)
        asr_model = EncDecCTCModel.restore_from(str(model_path))
        if not model_path.exists():
            possible_models = [m.stem for m in Path('nemo_models').iterdir()]
            raise FileNotFoundError(f'Model "{model_path.stem}" does not exist. Possible models: {possible_models}')
        for sample in samples:
            print('\nSegment and asr for ' + sample)
            torch.cuda.empty_cache()
            data_dir = Path('audio_samples', sample)
            audio_sample = Path(data_dir, 'main_audio.wav')

            # get model output
            log_prob, log_prob_step_sec, vocab, segments = segment_and_asr(str(audio_sample), asr_model, threshold, aggressiveness)

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
