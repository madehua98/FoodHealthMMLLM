# cli demo and quant to 4bit
conda activate llama_factory && export PYTHONPATH=/mnt/LLaMA-Factory && export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
conda install cudatoolkit-dev=11.7 -c conda-forge
pip install bitsandbytes --no-cache-dir
pip install optimum -U
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
pip install "fschat[model_worker,webui]"
pip install "gradio>=3.38.0,<4.0.0"
# install llvm
After cloning this repo, remove the .toml file
Go to setup.py file and set cuda version to 11.8, like;
MAIN_CUDA_VERSION = "11.8"
Then install xformers for cuda 11.8, with;
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
Comment out or remove, torch and xformers from the requirements.txt
You might get an error of packaging module not found, install with pip install packaging
Now you can install vLLM from source with CUDA 11.8 by simply running pip install -e .


# run server
python -m vllm.entrypoints.openai.api_server --model /media/fast_data/model/Meta-Llama-3-8B-Instruct --dtype half --max-model-len 1024
# inference using
tools/txt_filter_query.py