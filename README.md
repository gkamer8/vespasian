# Project Vespasian

This repo is for carrying out experiments related to training Llama from Facebook. The code uses low rank adaptation and 8 bit quantization to allow for efficient fine-tuning.

This project is based on [this repo](https://github.com/tloen/alpaca-lora), which is itself based on [Stanford's Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Setup

You should probably use a virtual environment, like so:

```
python3 -m venv venv
source venv/bin/activate
```

Then you'll need to install the requirements.

```
pip install -r requirements.txt
```

The requirements differ from the original tloen/alpaca-lora project because bitsandbytes should be built from source in most cases. It also clones from the most recent main branch of transformers, rather than from pip.

The most difficult part of the setup is dealing with the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) aspect, which allows for the 8 bit quantization. There's some info about compiling from source [here](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md).

In order to get that to work, you have to:

- clone the repo
- run `make cuda11x CUDA_VERSION=xxx` (or whichever for your version, which you can find using `nvcc --version`).
- you should use `find / -name libcudart.so 2>/dev/null` to find the LD_LIBRARY_PATH, which you then set.
- do `export LD_LIBRARY_PATH=...` with the elipses replaced by the previous path, which **should be a path to a directory with a libcudart.so file, not the file path itself**.
- still inside bitsandbytes, use `CUDA_VERSION=xxx python3 setup.py install`
- if working in a virtual environment, it'll need to be reactivated

## The Model

It should be possibly to refactor the code slightly to allow training alternative models, but this code currently is designed for Facebook's Llama 7b.

## Troubleshooting Notes

The original code used up about 19gb of vram on my graphics card; many errors can be caused by not having enough vram memory. Ideally you should use a cluster of 8 A100s but not everyone is so lucky.
