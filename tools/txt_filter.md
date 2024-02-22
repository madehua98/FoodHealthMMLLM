# cli demo and quant to 4bit
conda activate llama_factory && export PYTHONPATH=/mnt/LLaMA-Factory && export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
CUDA_VISIBLE_DEVICES=7 USE_MODELSCOPE_HUB=0 python src/cli_demo.py --model_name_or_path Qwen/Qwen1.5-14B-Chat --template qwen --finetuning_type lora  --quantization_bit 4 --temperature 0.1 --length_penalty 1.3
CUDA_VISIBLE_DEVICES=4,5,6 USE_MODELSCOPE_HUB=0 python -u src/export_model.py --model_name_or_path Qwen/Qwen1.5-14B-Chat --template qwen --export_dir /mnt/LLaMA-Factory/qwen1.5_14B_export_4bit --export_quantization_bit 4 --export_quantization_dataset data/c4_demo.json


# deploy with fastchat
nohup python3 -m fastchat.serve.controller > controller_qwen_filter.out &

CUDA_VISIBLE_DEVICES=7 nohup python3 -m fastchat.serve.vllm_worker --model-path Qwen/Qwen1.5-14B-Chat --limit-worker-concurrency 10 --num-gpus 1 --conv-template qwen > server_qwen_filter_fs.out &
CUDA_VISIBLE_DEVICES=5 nohup python3 -m fastchat.serve.vllm_worker --model-path qwen_14B_1228_export_4bit --limit-worker-concurrency 10 --conv-template qwen-7b-chat > server_qwen_14B_0104_export.out & 

nohup python3 -m fastchat.serve.openai_api_server --host localhost --port 8886 > server_qwen_filter.out &

# use api to query
