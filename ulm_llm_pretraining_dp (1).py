import numpy as np
import torch
import time
import os
from torch import nn
import math
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn import init
from transformers import AutoModelForCausalLM, AutoConfig
from ulm_dataloading import TokenizedDataSD, collate_function


def train(model, train_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device, dev_loader, test_loader, model_save_path):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 10 #in number of update steps (not num of batches)
    save_interval = 500
    start_time = time.time()

    num_batches = len(train_loader)
    optimizer.zero_grad()
    update_steps = 0
    batches_in_loss = 0
    for batch, (input_ids, labels, padding_mask) in enumerate(train_loader):
        batch_size = input_ids.shape[0]
        input_ids, labels, padding_mask = input_ids.to(device), labels.to(device), padding_mask.to(device)

        output = model(input_ids=input_ids, attention_mask=padding_mask)
        output = output['logits']

        output_flat = output.contiguous().view(-1, vocab_size)
        target_flat = labels.contiguous().view(-1)

        loss = criterion(output_flat, target_flat)
        total_loss += loss.item()
        batches_in_loss += 1

        loss /= grad_acc_step
        loss.backward()



        if (batch + 1) % grad_acc_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            update_steps += 1
            total_update_steps = (epoch * ( (num_batches + 1) // grad_acc_step) ) + update_steps

            if total_update_steps <= scheduler_config['warmup_steps']:
                optimizer.param_groups[0]['lr'] = scheduler_config['warmup_init'] + update_steps * scheduler_config['warmup_lr_step']
            else:
                #inverse square root learning schedule
                new_lr = scheduler_config['start_lr'] * (scheduler_config['warmup_steps'] ** 0.5)  * ( total_update_steps ** -0.5)
                optimizer.param_groups[0]['lr'] = new_lr


        ##logging condition
        if update_steps > 0 and update_steps % log_interval == 0 and (batch + 1) % grad_acc_step == 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time)
            cur_loss = total_loss / (batches_in_loss)
            print('denomoinator:', batches_in_loss, batch_size * grad_acc_step * log_interval)
            ppl = math.exp(cur_loss)

            #write logs
            f = open(logfile, 'a')
            f.write(f'| epoch {epoch:3d} | total_updates {total_update_steps:3d} | {batch:5d}/{num_batches:5d} batches | '
                f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.4f} | ppl {ppl:8.4f}\n')
            f.close()

            #reinitialize loss
            total_loss = 0
            batches_in_loss = 0
            start_time = time.time()

        ##saving and evaluating condition
        if update_steps > 0 and update_steps % save_interval == 0 and (batch + 1) % grad_acc_step == 0:
            print('saving model', batch, update_steps)
            
            model_save_location = model_save_path + 'epoch_' + str(epoch) + '_batch_' + str(batch)
            os.makedirs(model_save_location, exist_ok=True)
            original_model = model.module
            original_model.save_pretrained(model_save_location)

            evaluate(model, dev_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device)
            evaluate(model, test_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device)

            print('finished saving..')

    return model


def evaluate(model, data_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device):
    model.eval()  # turn on train mode
    total_loss = 0.
    log_interval = grad_acc_step * 10
    start_time = time.time()

    num_batches = len(data_loader)
    optimizer.zero_grad()

    with torch.no_grad():
        for batch, (input_ids, labels, padding_mask) in enumerate(data_loader):
            input_ids, labels, padding_mask = input_ids.to(device), labels.to(device), padding_mask.to(device)

            output = model(input_ids=input_ids, attention_mask=padding_mask)
            output = output['logits']

            output_flat = output.contiguous().view(-1, vocab_size)
            target_flat = labels.contiguous().view(-1)

            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()


    lr = optimizer.param_groups[0]['lr']
    ms_per_batch = (time.time() - start_time)
    cur_loss = total_loss / num_batches
    ppl = math.exp(cur_loss)
    #write logs
    f = open(logfile, 'a')
    f.write(f'EVAL | epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            f'lr {lr:02.8f} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}\n')
    f.close()

def apply_xavier_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def main():

    #load model
    start_epoch = 0
    
    #logging
    model_name = 'facebook/opt-125m'
    ulm_path = model_name#'/data/akshat/twist_finetuned/sdhubert_ls_facebook-opt-125m/epoch_0_final'
    data_location = None#'/home/akshatgupta/bsg_textless/twist_datasets/ls-hubert-base-ls960-100/'#end location with /
    train_type = 'sdhubert_original_weights_ls_' + model_name.replace('/','-')
    ulm_offset = 2
    ulm_vocab_size = 4096
    ###Edit above things - start epoch, ulm_path (starting model path), dataset location and train type, and vocab size
    
    logfile = 'training_logs/logs_' + train_type + '.txt'
    model_save_path = '/data/akshat/twist_finetuned/' + train_type + '/'
    os.makedirs(model_save_path, exist_ok=True)

    config = AutoConfig.from_pretrained(ulm_path)
    model = AutoModelForCausalLM.from_pretrained(ulm_path)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num of parameter:', params/1e6, 'M')

    #reinitialize embedding layer
    vocab_size = ulm_vocab_size + ulm_offset
    embedding_size = config.word_embed_proj_dim

    if start_epoch == 0:
        model.config.vocab_size = vocab_size
        model.config.word_embed_proj_dim = embedding_size

        #assign new embedding layer
        model.model.decoder.embed_tokens = torch.nn.Embedding(vocab_size, embedding_size)
        model.lm_head = torch.nn.Linear(embedding_size, vocab_size, bias=False)

        # Apply Xavier initialization to all layers
        model.apply(apply_xavier_init)

        #tie weights
        model.tie_weights()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('New Num of parameter:', params/1e6, 'M')

    #params
    max_tokens = config.max_position_embeddings
    padding_idx = config.pad_token_id
    token_offset = ulm_offset
    batch_size = 8
    grad_acc_step = 8
    n_epochs= 100


    #move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    #training parameters
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    #define lr scheduler config file
    scheduler_config = {
        'warmup_init': 1e-07,
        'start_lr': 5e-4, # from paper
        'final_lr': 8e-5,
        'warmup_steps': 4000,
        'linear_steps': 5000,
    }
    scheduler_config['warmup_lr_step'] = ( scheduler_config['start_lr'] - scheduler_config['warmup_init'] ) / scheduler_config['warmup_steps']
    scheduler_config['lr_step'] = ( scheduler_config['start_lr'] - scheduler_config['final_lr']) / scheduler_config['linear_steps']##For linear scheduler

    #define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = scheduler_config['warmup_init'], betas=(0.9, 0.95), weight_decay = 0.1)

    #dataset
    train_dataset = TokenizedDataSD(dataset_type = 'train', max_tokens=max_tokens, token_offset=token_offset, padding_idx=padding_idx, data_location=data_location)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, collate_fn=collate_function)
    dev_dataset = TokenizedDataSD(dataset_type = 'dev', max_tokens=max_tokens, token_offset=token_offset, padding_idx=padding_idx, data_location=data_location)
    dev_loader = DataLoader(dev_dataset, batch_size = batch_size , collate_fn=collate_function)
    test_dataset = TokenizedDataSD(dataset_type = 'test', max_tokens=max_tokens, token_offset=token_offset, padding_idx=padding_idx, data_location=data_location)
    test_loader = DataLoader(test_dataset, batch_size = batch_size , collate_fn=collate_function)

    print('TRAIN BATCHES:', len(train_loader), 'DEV BATCHES:', len(dev_loader), 'TEST BATCHES:', len(test_loader))
    for epoch in range(start_epoch, n_epochs):
        model = train(model, train_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device, dev_loader, test_loader, model_save_path)

        #save model
        print('saving model after epoch', epoch)
        model_save_location = model_save_path + 'epoch_' + str(epoch) + '_final'
        os.makedirs(model_save_location, exist_ok=True)
        original_model = model.module
        original_model.save_pretrained(model_save_location)
        print('Finished final saving')

        #evaluate model
        evaluate(model, dev_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device)
        evaluate(model, test_loader, criterion, optimizer, scheduler_config, vocab_size, epoch, grad_acc_step, logfile, device)


if __name__ == '__main__':
    main()

