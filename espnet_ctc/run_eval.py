import argparse
import evaluate
import librosa
import os
import torch
from tqdm import tqdm
from datasets import Dataset

from normalizer import data_utils
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


def dataset_iterator(dataset: Dataset):
    """Iterate over the dataset and yield the audio and reference text.

    Arguments
    ---------
    dataset : Dataset
        The dataset to iterate over.

    Yields
    ------
    dict
        A dictionary containing the audio and reference text.
    """
    for i, item in enumerate(dataset):
        yield {
            **item["audio"],    # "path", "array"
            "reference": item["norm_text"],
            # "audio_filename": f"file_{i}",
            # "sample_rate": 16_000,
            # "sample_id": i,
        }


def main(args):
    """Run the evaluation script."""

    assert args.batch_size == 1

    if args.device == -1:
        device = "cpu"
    else:
        device = f"cuda:{args.device}"

    model = Speech2TextGreedySearch.from_pretrained(
        model_tag=args.model_tag,
        device=device,
        # below are default values which can be overwritten in __call__
        lang_sym="<eng>",
        task_sym="<asr>",
    )

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    predictions = []
    references = []
    for sample in tqdm(
        dataset_iterator(dataset),
        desc="Evaluating: Sample id",
        unit="",
        disable=False,
    ):
        references.append(sample["reference"])
        speech = librosa.util.fix_length(sample["array"], size=(16000 * 30))
        with autocast(**autocast_args):
            predictions.append(
                data_utils.normalizer(
                    model(speech)[0][-2]
                )
            )

    # Write manifest results
    manifest_path = data_utils.write_manifest(
        references, predictions, args.model_tag, args.dataset_path, args.dataset, args.split
    )
    print("Results saved at path:", os.path.abspath(manifest_path))
    
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_tag",
        type=str,
        required=True,
        help="ESPnet model tag",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args)
