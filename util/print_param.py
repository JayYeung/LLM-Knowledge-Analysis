from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from attrdict import AttrDict

import nethook


if __name__ == '__main__':
    top_k = 5
    #model_name = '/data/akshat/models/Llama-2-7b-hf'
    model_name = "allenai/OLMo-7B"
    #hparams_filename = 'llama2-7b.json'
    #model_name = 'gpt2-xl'
    #hparams_filename = 'gpt2-xl.json'

    #f = open(hparams_filename)
    #hparams = AttrDict(json.load(f))

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for n, m in model.named_modules():
        print('N', n)
        print('M', m)
        print()