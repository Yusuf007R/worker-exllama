import glob
import logging
import os
from copy import copy

import runpod
from generator import ExLlamaGenerator
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

model_directory = './model'
print(model_directory)
tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
st_files = glob.glob(st_pattern)
print(st_files)
if not st_files:
    raise ValueError(f"No safetensors files found in {model_directory}")
model_path = st_files[0]
# Create config, model, tokenizer and generator
config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file
model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
default_settings = {
    k: getattr(generator.settings, k) for k in dir(generator.settings) if k[:2] != '__'
}

def generate_with_streaming(prompt, max_new_tokens):
    generator.end_beam_search()

    # Tokenizing the input
    ids = generator.tokenizer.encode(prompt)
    ids = ids[:, -generator.model.config.max_seq_len:]

    generator.gen_begin_reuse(ids)
    initial_len = generator.sequence[0].shape[0]
    has_leading_space = False
    for i in range(max_new_tokens):
        token = generator.gen_single_token()
        if i == 0 and generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
            has_leading_space = True

        decoded_text = generator.tokenizer.decode(generator.sequence[0][initial_len:])
        if has_leading_space:
            decoded_text = ' ' + decoded_text

        yield decoded_text
        if token.item() == generator.tokenizer.eos_token_id:
            break
        
        

def inference(event):
    logging.info(event)
    job_input = event["input"]
    if not job_input:
        raise ValueError("No input provided")

    prompt: str = job_input.pop("prompt")
    max_new_tokens = job_input.pop("max_new_tokens", 50)
    human_prefix = job_input.pop("human_prefix", "USER")
    generator_settings = job_input.pop("generator_settings", {})
  

    settings = copy(default_settings)
    settings.update(generator_settings)
    for key, value in settings.items():
        setattr(generator.settings, key, value)

    output = ''
    for output in generate_with_streaming(prompt, max_new_tokens):
        if output.endswith(f"{human_prefix}:"):
            output = output[len(human_prefix):]
            break

    yield output

runpod.serverless.start({"handler": inference})
