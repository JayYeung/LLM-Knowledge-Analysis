from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import torch.nn.functional as F
from attrdict import AttrDict
from datasets import load_dataset
import random
import os
import util.nethook as nethook
from util.tok_dataset import TokenizedDataset, dict_to_
from util.get_trace_layers import get_trace_layers
from dsets.counterfact import CounterFactDataset
from dsets.mquake import MQuAKEPromptCompletionDataset
from dsets.part_of_sentence import PromptCompletionDataset
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from tqdm import tqdm, trange
import pandas as pd

random.seed(42)

if __name__ == '__main__':
    # SELECT THE MODEL HERE

    model_name = '/data/akshat/models/gpt2-xl'
    hparams_filename = 'hparams/gpt2-xl.json'
    
    # model_name = "/data/akshat/models/Llama-2-7b-hf"
    # hparams_filename = 'hparams/llama2-7b.json'  
    
    # model_name = "/data/akshat/models/pythia-6.9b"
    # hparams_filename = 'hparams/pythia-6.9.json'
    
    # model_name = "allenai/OLMo-7B"
    # hparams_filename = 'hparams/olmo.json'

    model_filename = model_name.split('/')[-1]

    # load params file
    f = open(hparams_filename)
    hparams = AttrDict(json.load(f))
    
    if model_filename == "OLMo-7B": 
        model = OLMoForCausalLM.from_pretrained(model_name, cache_dir="/data/akshat/models/").cuda()
        tokenizer = OLMoTokenizerFast.from_pretrained(model_name, cache_dir="/data/akshat/models/")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    #get necessary layers
    trace_layers, ln_1_layers, ln_2_layers, lm_head, ln_f, n_layers = get_trace_layers(model, hparams)
    
    model_output_name = model_filename
    skip_tokens = 0
    if model_filename == 'gpt2-xl':
        embedding_matrix = nethook.get_module(model, "transformer.wte")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.transformer.wte(input_ids)  
    elif model_filename == 'Llama-2-7b-hf':
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Llama-2-7b'
        skip_tokens = 1
    elif model_filename == 'pythia-6.9b':
        embedding_matrix = nethook.get_module(model, "embed_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.gpt_neox.embed_in(input_ids)  
    else:
        embedding_matrix = nethook.get_module(model, "model.transformer.ff_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.get_input_embeddings()(input_ids)
        model_output_name = 'Olmo'
    
    # number of non-space tokens we want to predict
    TOKENS_PREDICTED = 1

    for pos_tag in ['REASONING']: 
        if pos_tag == 'FACT':
            dataset = CounterFactDataset('data', multi=False)
        elif pos_tag == 'REASONING':
            dataset = MQuAKEPromptCompletionDataset(max_examples=30_0)
        elif pos_tag == 'QNA':
            dataset = MQuAKEPromptCompletionDataset(type='qna', max_examples=30_000)
        else:
            dataset = PromptCompletionDataset(pos_tag=pos_tag, min_prompt_length=17, max_examples = 30_000)

        output_df = []

        for i in trange(len(dataset)):
            item = dataset.__getitem__(i)
            # Prepare prompt and answer as before
            if pos_tag == 'FACT':
                prompt = item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject'])
                answer = item['requested_rewrite']['target_true']['str']
            else:
                prompt = item['prompt']
                answer = item['answer']
            
            answer_token = tokenizer.encode(answer, return_tensors='pt').cuda()[0]
            answer_token_w_space = tokenizer.encode(' ' + answer, return_tensors='pt').cuda()[0]
            first_answer_token = answer_token[skip_tokens]
            first_answer_token_w_space = answer_token_w_space[skip_tokens]
            
            def run_model(prompt, noise_layer=None):
                input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
                
                handles = []
                if noise_layer is not None:
                    noise_level = 3
                    def add_noise_hook(module, input, output):
                        if isinstance(output, tuple):
                            noisy_output = tuple(
                                o + torch.randn_like(o) * noise_level if torch.is_tensor(o) else o
                                for o in output
                            )
                            return noisy_output
                        elif torch.is_tensor(output):
                            noisy_output = output + torch.randn_like(output) * noise_level
                            return noisy_output
                        else:
                            return output

                    for name, module in model.named_modules():
                        if name == noise_layer:
                            handle = module.register_forward_hook(add_noise_hook)
                            handles.append(handle)
                            break  

                with nethook.TraceDict(
                    module=model,
                    layers=trace_layers,
                    retain_input=True,
                    retain_output=False,
                ) as tr:
                    outputs = model.generate(
                        input_ids, 
                        max_new_tokens=1, 
                        num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id,
                        output_scores=True,  
                        return_dict_in_generate=True,  
                    )

                for handle in handles:
                    handle.remove()
                
                output_ids = outputs.sequences
                last_token = output_ids[0][-1]
                output_text = tokenizer.decode(last_token, skip_special_tokens=True)

                logits = outputs.scores[0]  

                return output_text, last_token, logits

            output_text, last_token, baseline_logits = run_model(prompt)

            if last_token != first_answer_token and last_token != first_answer_token_w_space:
                continue  

            with torch.no_grad():
                baseline_probs = torch.softmax(baseline_logits, dim=-1)
                baseline_prob = baseline_probs[0, last_token]

            for noise_layer in trace_layers:
                output_text_noisy, last_token_noisy, noisy_logits = run_model(
                    prompt, noise_layer=noise_layer
                )
                
                with torch.no_grad():
                    noisy_probs = torch.softmax(noisy_logits, dim=-1)
                    last_token_prob = noisy_probs[0, last_token]
                
                prob_change = last_token_prob - baseline_prob
                output_df.append({
                    'layer': noise_layer,
                    'prompt index': i,
                    'prompt': prompt,
                    'probability': last_token_prob.item(),
                    'prob_change': prob_change.item()
                })
                # print(noise_layer + ' -> ' + output_text_noisy + ' -> ' + str(round(prob_change.item(), 2)))



        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{model_output_name}_causal_small_{pos_tag}.csv', index = False)