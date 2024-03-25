import time
import librosa
import torch

from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

autocast_args = dict(
    enabled=True
)
from torch.cuda.amp import autocast
if (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
):
    autocast_args = dict(dtype=torch.bfloat16)

device = "cuda:0"

models = [
    "pyf98/owsm_ctc_v3.1_1B",
]

n_batches = 3
warmup_batches = 5

audio_file = "../data/sample_4469669.wav"
max_len = 600  # 10 minutes

batch_size = 32     # batched parallel decoding; it depends on the GPU VRAM


def pre_process_audio(audio_file, sr, max_len):
    _, _sr = librosa.load(audio_file, sr=sr)
    audio_len = int(max_len * _sr)
    audio_arr = _[:audio_len]
    return {"raw": audio_arr, "sampling_rate": _sr}, audio_len


audio_dict, audio_len = pre_process_audio(audio_file, 16000, max_len)

rtfs = []

for model in models[:1]:
    model = Speech2TextGreedySearch.from_pretrained(
        model_tag=model,
        device=device,
        dtype="float32",
        # below are default values which can be overwritten in __call__
        lang_sym="<eng>",
        task_sym="<asr>",
    )

    for i in range(3):
        print(f"outer_loop -> {i}")
        total_time = 0.0
        for _ in range(n_batches + warmup_batches):
            print(f"batch_num -> {_}")
            start = time.time()
            with autocast(**autocast_args):
                text = model.decode_long_batched_buffered(
                    audio_dict["raw"],
                    batch_size=batch_size,
                    context_len_in_secs=2,
                    frames_per_sec=12.5,        # 80ms shift, model-dependent, don't change
                )
            end = time.time()
            if _ >= warmup_batches:
                total_time += end - start

        rtf = (total_time / n_batches) / (audio_len / 16000)
        rtfs.append(rtf)

    print(f"all RTFs: {model}: {rtfs}")
    rtf_val = sum(rtfs) / len(rtfs)
    print(f"avg. RTF: {model}: {rtf_val}")
