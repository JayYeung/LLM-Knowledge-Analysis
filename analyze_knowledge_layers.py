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
from tuned_lens.nn.lenses import TunedLens, LogitLens

random.seed(42)

if __name__ == '__main__':
    # SELECT THE MODEL HERE

    model_name = '/data/akshat/models/gpt2-xl'
    hparams_filename = 'hparams/gpt2-xl.json'
    
    model_name = "/data/akshat/models/Llama-2-7b-hf"
    hparams_filename = 'hparams/llama2-7b.json'  
    
    model_name = "/data/akshat/models/pythia-6.9b"
    hparams_filename = 'hparams/pythia-6.9.json'
    
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
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = model_filename).cuda()
        embedding_matrix = nethook.get_module(model, "transformer.wte")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.transformer.wte(input_ids)  
    elif model_filename == 'Llama-2-7b-hf':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'meta-llama/' + model_filename).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Llama-2-7b'
        skip_tokens = 1
    elif model_filename == 'pythia-6.9b':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'EleutherAI/pythia-6.9b-deduped').cuda()
        embedding_matrix = nethook.get_module(model, "embed_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.gpt_neox.embed_in(input_ids)  
    else:
        tuned_lens = None
        embedding_matrix = nethook.get_module(model, "model.transformer.ff_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.get_input_embeddings()(input_ids)
        model_output_name = 'Olmo'
    
    print(f'Running model {model_output_name}')
    
    USING_TUNED = True
    PREVIOUS_TOKEN = True

    for pos_tag in ['REASONING']: 
        if pos_tag == 'FACT':
            dataset = CounterFactDataset('data', multi=False)
        elif pos_tag == 'REASONING':
            dataset = MQuAKEPromptCompletionDataset(max_examples=30_000)
        elif pos_tag == 'QNA':
            dataset = MQuAKEPromptCompletionDataset(type='qna', max_examples=30_000)
        elif pos_tag == 'MULTIQNA':
            dataset = MQuAKEPromptCompletionDataset(type='multiqna', max_examples=30_000)
        else:
            dataset = PromptCompletionDataset(pos_tag=pos_tag, min_prompt_length=17, max_examples = 30_000)

        output_df = []
        second_token_df = []
        third_token_df = []

        for i in trange(len(dataset)):
            item = dataset.__getitem__(i)
            if pos_tag == 'FACT':
                prompt = item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject'])
                answer = item['requested_rewrite']['target_true']['str']

            else:
                prompt = item['prompt']
                answer = item['answer']
                
            if PREVIOUS_TOKEN and prompt:
                prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').cuda()[0]
                last_prompt_token = prompt_tokens[-1].unsqueeze(0)
                prompt = tokenizer.decode(prompt_tokens[:-1]).strip()
                
                if len(prompt) < 5:
                    continue
            
                            
            if answer is not None: 
                answer_token = tokenizer.encode(answer, return_tensors='pt').cuda()[0]
                answer_token_w_space = tokenizer.encode(' ' + answer, return_tensors='pt').cuda()[0]
                
                answer_token_length = len(answer_token)
                answer_token_w_space_length = len(answer_token_w_space)

                first_answer_token = answer_token[skip_tokens]
                first_answer_token_w_space = answer_token_w_space[skip_tokens]

                second_answer_token = answer_token[skip_tokens + 1] if answer_token_length > skip_tokens + 1 else None
                second_answer_token_w_space = answer_token_w_space[skip_tokens + 1] if answer_token_w_space_length > skip_tokens + 1 else None

                third_answer_token = answer_token[skip_tokens + 2] if answer_token_length > skip_tokens + 2 else None
                third_answer_token_w_space = answer_token_w_space[skip_tokens + 2] if answer_token_w_space_length > skip_tokens + 2 else None

            def run_model(prompt): 
                input_ids = tokenizer.encode(prompt, return_tensors='pt', 
                                        truncation=True, max_length=1024).cuda()
                with nethook.TraceDict(
                    module=model,
                    layers=trace_layers,
                    retain_input=True,
                    retain_output=False,
                ) as tr:                    
                    output_ids = model.generate(
                        input_ids, 
                        max_new_tokens=1, 
                        num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                last_token = output_ids[0][-1]
                output_text = tokenizer.decode(last_token, skip_special_tokens=True)
                input_embeddings = get_input_embeddings(input_ids)
                first_token_embedding = input_embeddings[:, -1, :][0]
                return output_text, last_token, tr, first_token_embedding
            
            def parse_data(last_token, tr, first_token_embedding, result_df, tuned=False): 
                last_embed = None
                # print(f'prompt: {prompt} answer: {answer}')
                # After generation, you can access the traced data
                for layer_name in tr:
                    with torch.no_grad():
                        cur_in = tr[layer_name].input  
                        if tuned and tuned_lens is not None and layer_name not in ('transformer.ln_f', 'model.norm', 'gpt_neox.final_layer_norm'):
                            layer = int(layer_name.split('.')[-1])
                            h = cur_in[0][-1]
                            cur_out = tuned_lens(h, layer)
                        else:
                            
                            final_ln_output = ln_f(cur_in) 
                            h = final_ln_output[0][-1]  # [4096]
                            cur_out = embedding_matrix.weight @ h  # [32000, 4096] x [4096]
                        
                        probabilities = torch.softmax(cur_out, dim=0)
                        token_id = torch.argmax(cur_out).item()
                        last_token_prob = probabilities[last_token].item()
                        rank = torch.sum(probabilities > last_token_prob).item() + 1
                    
                        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-50))
                        average_entropy = float(torch.mean(entropy))

                        guess = tokenizer.decode([token_id], skip_special_tokens=True)
                        angle_diff = None
                        norm_diff = None
                                                
                        if last_embed is not None:
                            cos_sim = F.cosine_similarity(h, last_embed, dim=0)
                            angle_diff = torch.acos(cos_sim.clamp(-1.0, 1.0)).item()  
                            norm_diff = torch.norm(h - last_embed).item()

                        cos_sim_w_first = F.cosine_similarity(h, first_token_embedding, dim=0)
                        angle_diff_w_first = torch.acos(cos_sim_w_first.clamp(-1.0, 1.0)).item()  
                        norm_diff_w_first = torch.norm(h - first_token_embedding).item()

                        last_embed = h.clone()
                        
                        result_df.append({
                            'layer': layer_name, 
                            'prompt index': i, 
                            'prompt': prompt, 
                            'answer': int(last_token), 
                            'answer_text' : answer, 
                            'guess': token_id, 
                            'rank': rank, 
                            'probability': last_token_prob, 
                            'confidence': average_entropy, 
                            'angle_diff': angle_diff, 
                            'norm_diff': norm_diff,
                            'angle_diff_w_first' : angle_diff_w_first, 
                            'norm_diff_w_first' : norm_diff_w_first
                        })
            
            output_text, last_token, tr, first_token_embedding = run_model(prompt)
            if output_text.isspace(): 
                prompt += ' '
                output_text, last_token, tr, first_token_embedding = run_model(prompt)
                if output_text.isspace():
                    continue

            if PREVIOUS_TOKEN:
                parse_data(first_answer_token, tr, first_token_embedding, output_df)
                continue
            
            if answer is not None and last_token != first_answer_token and last_token != first_answer_token_w_space:
                continue
            parse_data(last_token, tr, first_token_embedding, output_df, tuned=USING_TUNED)
            
            prompt += output_text
            output_text, last_token, tr, first_token_embedding = run_model(prompt)
            if last_token != second_answer_token and last_token != second_answer_token_w_space: 
                continue
            parse_data(last_token, tr, first_token_embedding, second_token_df, tuned=USING_TUNED)

            prompt += output_text
            output_text, last_token, tr, first_token_embedding = run_model(prompt)
            if last_token != third_answer_token and last_token != third_answer_token_w_space: 
                continue
            parse_data(last_token, tr, first_token_embedding, third_token_df, tuned=USING_TUNED)
        
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{model_output_name}_{pos_tag}_tuned_-1.csv', index = False)

        # second_token_df = pd.DataFrame(second_token_df)
        # second_token_df.to_csv(f'{model_output_name}_{pos_tag}_second_tuned.csv', index = False)
        
        # third_token_df = pd.DataFrame(third_token_df)
        # third_token_df.to_csv(f'{model_output_name}_{pos_tag}_third_tuned.csv', index = False)