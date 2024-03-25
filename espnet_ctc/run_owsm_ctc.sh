#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_TAGs=("pyf98/owsm_ctc_v3.1_1B")
BATCH_SIZE=1

num_models=${#MODEL_TAGs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_TAG=${MODEL_TAGs[$i]}

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_tag=${MODEL_TAG} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="common_voice" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_TAG}')" && \
    cd $RUNDIR

done
