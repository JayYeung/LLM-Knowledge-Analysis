#here we create 10 samples of size 500 to sample the counterfact dataset from to visualize model editing effects

from dsets.counterfact import CounterFactDataset
import random
import json

random.seed(37)

if __name__ == '__main__':

    num_samples = 10 #number of samples
    sample_size = 20000
    mcf_flag = False

    dataset = CounterFactDataset('data', multi=mcf_flag)

    ####select unique subjects
    all_subjects = {}

    subject2rel = []
    selected_indices = []
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        
        found_flag = False
        relation = item['requested_rewrite']['relation_id'].lower()
        subject = item['requested_rewrite']['subject'].lower()
        target_new = item['requested_rewrite']['target_new']['str'].lower()
        target_true = item['requested_rewrite']['target_true']['str'].lower()

        basic_prompt = item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject'])

        print(f'{basic_prompt = }')

        if i == 10:
            break

        if subject in all_subjects:
            continue 
        else:
            all_subjects[subject] = None

        #if reached here, keep example            
        selected_indices.append(i)


    sampled_indices = {}
    for n in range(num_samples):
        random.shuffle(selected_indices)
        sampled_indices[n] = selected_indices[:sample_size]

