<<<<<<< HEAD
#!/bin/bash


CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
deepspeed moellava/eval/model_vqa_science.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_science_qa.py \
    --base-dir ${EVAL}/scienceqa \
    --result-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}_output.jsonl \
=======
#!/bin/bash


CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
deepspeed moellava/eval/model_vqa_science.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_science_qa.py \
    --base-dir ${EVAL}/scienceqa \
    --result-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}_output.jsonl \
>>>>>>> upstream/main
    --output-result ${EVAL}/scienceqa/answers/${CKPT_NAME}_result.json