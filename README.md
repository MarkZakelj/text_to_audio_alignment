# Text to audio alignment

## Env creation
* download and build kenlm binaries from [github](https://github.com/kpu/kenlm)
* use `apt-get update && apt-get install -y libsndfile1 ffmpeg` (nemo dependencies)  
* create conda env: `conda create --file conda_env.yml`  
* within new env, install `pytorch 1.8.1 LTS` (CUDA 10 or CPU version), from [pytorch downloads](https://pytorch.org/get-started/locally/)

## Usage
* set config file `config.txt`  
* run `main_segment_and_asr.py` to generate segments and model output (CTC)
* run `main.py` to decode CTC, align and eval alltogether OR: run `main_decode.py` then `main_align.py` then `eval.py`

## File structure
```
audio_samples\  
+-- <sample_name>\  
|   +-- main_text.txt (required)  
|    +-- main_audio.wav (required)  
|    +-- alignment_reference.json (not required for usage, only for evaluation)  
|    +-- results\ (auto generated)  
|     +-- <model_name>\  
|        +-- output.npy  
|        +-- <ctc_mode>\
|          +-- transcript.txt     
|          +-- char_ms.txt  
|          +-- aligned_phrases.txt  
|          +-- aligned_words.txt  
|          +-- eval_results.txt
```