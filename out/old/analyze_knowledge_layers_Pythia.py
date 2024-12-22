from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoXForCausalLM, AutoTokenizer
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
from dsets.part_of_sentence import PromptCompletionDataset
from dsets.mquake import MQuAKEPromptCompletionDataset
from tqdm import tqdm, trange
import pandas as pd

random.seed(42)

if __name__ == '__main__':
    model_name = "/data/akshat/models/pythia-6.9b"
    hparams_filename = 'hparams/pythia-6.9.json'

    with open(hparams_filename) as f:
        hparams = AttrDict(json.load(f))

    model = GPTNeoXForCausalLM.from_pretrained(model_name).cuda()  # Using LLaMA-specific model class
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #get necessary layers
    trace_layers, ln_1_layers, ln_2_layers, lm_head, ln_f, n_layers = get_trace_layers(model, hparams)

    embedding_matrix = nethook.get_module(model, "embed_out")

    for pos_tag in ['REASONING']: 
        if pos_tag == 'FACT':
            dataset = CounterFactDataset('data', multi=False)
        elif pos_tag == 'REASONING':
            dataset = MQuAKEPromptCompletionDataset(max_examples=30_000)
        else:
            dataset = PromptCompletionDataset(pos_tag=pos_tag, min_prompt_length=17, max_examples = 30_000)

        output_df = []

        for i in trange(len(dataset)):
            item = dataset.__getitem__(i)
            if pos_tag == 'FACT':
                prompt = item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject'])
                answer = item['requested_rewrite']['target_true']['str']

            else:
                prompt = item['prompt']
                answer = item['answer']
            
            # print(f'{prompt = } {answer = }')

            input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
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

            answer_token = tokenizer.encode(answer, return_tensors='pt').cuda()
            answer_token_w_space = tokenizer.encode(' ' + answer, return_tensors='pt').cuda()
            last_token = output_ids[0][-1]
            output_text = tokenizer.decode(last_token, skip_special_tokens=True)
            # model eventually gets correct answer
            # take last token is FIRST token of `answer`


            print('answer', answer)
            print('answer_token', answer_token)
            print('answer_token_w_space', answer_token_w_space)
            print()

# answer London
# answer_token tensor([[18868]], device='cuda:0')
# answer_token_w_space tensor([[4693]], device='cuda:0')

            if last_token not in answer_token and last_token not in answer_token_w_space: 
                continue

            with torch.no_grad():
                input_embeddings = model.gpt_neox.embed_in(input_ids)  

            first_token_embedding = input_embeddings[:, -1, :][0]

            last_embed = None

            # After generation, you can access the traced data
            for layer_name in tr:
                with torch.no_grad():
                    cur_in = tr[layer_name].input  
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

                    output_df.append({
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
            # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # print(f'{final_answer = }')

        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'pythia-6.9b_{pos_tag}.csv', index = False)

