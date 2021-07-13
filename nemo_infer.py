import json
import tempfile
import os

import scipy.io as sio
from nemo.collections.asr.models import EncDecCTCModel
import torch


def infer(model, audiofiles, batch_size=4):
    if isinstance(model, str):
        asr_model = EncDecCTCModel.restore_from(model)
    else:
        asr_model = model

    mode = asr_model.training
    device = next(asr_model.parameters()).device
    # device = torch.device("cpu")
    asr_model.eval()
    vocab=asr_model._cfg.train_ds.labels
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
            for file in audiofiles:
                entry = {'audio_filepath': file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': audiofiles, 'batch_size': batch_size, 'temp_dir': tmpdir}

        characters=[]
        log_probs=[]
        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        for test_batch in temporary_datalayer:
            log_prob, encoded_len, greedy_predictions = asr_model.forward(input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device))
            character = asr_model._wer.ctc_decoder_predictions_tensor(greedy_predictions)
            characters += character
            encoded_len = encoded_len.long().cpu()
            log_prob = log_prob.float().cpu()
            for i in range(0,encoded_len.shape[0]):
                el = encoded_len[i].detach().numpy().tolist()
                lp = log_prob[i].detach().numpy().tolist()

                log_probs+=[lp[0:el]]
            del test_batch

    asr_model.train(mode)
    return characters, log_probs, vocab


def main():
    model='/home/FRI1/matic/Research/Speech/Recognizer/nemo_experiments/QuartzNet5x3/2020-09-09_11-04-36/checkpoints/QuartzNet5x3--last.ckpt'

    eval_file="/home/FRI1/matic/Research/Databases/Speech/Gos/Gos.wav/JIfasvakdr-rd1001290912_s2_1746.056.wav"
    eval_trans="ta zadn del ki je kompliciran je izhaja iz zakona o prejemu državljanstva"

    characters, log_prob, vocab = infer(model, [eval_file])

    sio.savemat('output.mat', {'model': model, 'log_prob': log_prob, 'characters': characters, 'truth': eval_trans, 'eval_file': eval_file, 'vocab': vocab})

    print(characters)

if __name__ == '__main__':
    main()
