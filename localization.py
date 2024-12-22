import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    AdamW,
    get_linear_schedule_with_warmup,
)
from dsets.counterfact import CounterFactDataset
from dsets.part_of_sentence import PromptCompletionDataset

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def compute_loss(outputs, labels, model, hparams, additional_inputs=None):
    """
    Custom loss function that includes an additional loss term based on the outputs from an earlier layer.
    
    Args:
        outputs: Model outputs from the forward pass.
        labels: Ground truth labels with -100 for tokens to ignore.
        model: The GPT2 model.
        hparams: Hyperparameters dictionary.
        additional_inputs: Any additional inputs needed for the custom loss term.
        
    Returns:
        total_loss: The combined loss value.
    """
    
    # Standard CrossEntropyLoss
    logits = outputs.logits  
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    ce_loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    # Additional loss term from earlier layer
    layer_index = hparams.get("first_layer", 10)
    hidden_states = outputs.hidden_states 
    hidden_state = hidden_states[layer_index] 
    
    # Pass through final layer norm
    final_layer_norm = model.transformer.ln_f
    final_ln_output = final_layer_norm(hidden_state) 
    
    # Compute logits using the embedding matrix
    embedding_matrix = model.transformer.wte
    logits_additional = F.linear(final_ln_output, embedding_matrix.weight)  

    # Compute additional cross-entropy loss
    additional_loss = loss_fct(
        logits_additional.view(-1, logits_additional.size(-1)),
        labels.view(-1)
    )
    
    # Combine losses with a weighting factor
    additional_loss_weight = hparams.get("additional_loss_weight", 1.0)
    total_loss = ce_loss + additional_loss_weight * additional_loss
    return total_loss


def load_hparams(hparams_filename: str):
    with open(hparams_filename, 'r') as f:
        hparams = json.load(f)
    return hparams

def main():
    # Paths
    model_name = "/data/akshat/models/gpt2-xl"
    hparams_filename = "hparams/gpt2-xl.json"
    data_dir = 'data'  # Update with your data directory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load hyperparameters
    hparams = load_hparams(hparams_filename)

    # Extract relevant hyperparameters
    batch_size = hparams.get("batch_size", 1)
    epochs = hparams.get("epochs", 3)
    learning_rate = hparams.get("learning_rate", 5e-5)
    gradient_accumulation_steps = hparams.get("gradient_accumulation_steps", 1)
    max_grad_norm = hparams.get("max_grad_norm", 1.0)
    warmup_steps = hparams.get("warmup_steps", 100)
    logging_steps = hparams.get("logging_steps", 100)
    save_steps = hparams.get("save_steps", 1000)
    output_dir = hparams.get("output_dir", "./finetuned_gpt2")

    # Load configuration, tokenizer, and model
    config = GPT2Config.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()  # Set model to training mode

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)

    # Prepare dataset
    train_dataset = PromptCompletionDataset(pos_tag='DET', min_prompt_length=17, max_examples = 30_000)

    def collate_fn(batch):
        input_texts = []
        labels = []
        for item in batch:
            prompt = item['prompt']
            answer = item['answer']
            combined_text = prompt + answer
            input_texts.append(combined_text)

            # Tokenize the combined text
            encoding = tokenizer(
                combined_text,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            input_ids = encoding.input_ids[0]

            # Create labels with -100 for prompt tokens and actual IDs for answer tokens
            prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            labels_for_example = [-100] * prompt_length + input_ids[prompt_length:].tolist()
            labels.append(labels_for_example)

        # Pad input_ids and labels to the same length
        max_length = max(len(ids) for ids in input_texts)

        # Tokenize inputs with padding
        inputs = tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Pad labels
        labels_padded = []
        for l in labels:
            l = l + [-100] * (inputs.input_ids.shape[1] - len(l))
            labels_padded.append(l)
        labels_tensor = torch.tensor(labels_padded)

        return inputs.input_ids, inputs.attention_mask, labels_tensor

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Define scheduler
    total_steps = (
        len(train_dataloader) // gradient_accumulation_steps
    ) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    global_step = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            inputs, attention_mask, labels = batch  # Unpack three values
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask, output_hidden_states=True)

            # Compute loss with labels
            loss = compute_loss(outputs, labels, model, hparams)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    print(
                        f"Epoch {epoch} Global Step {global_step} Loss {loss.item() * gradient_accumulation_steps:.4f}"
                    )

                if global_step % save_steps == 0:
                    # Save model checkpoint
                    output_dir_checkpoint = os.path.join(
                        output_dir, f"checkpoint-{global_step}"
                    )
                    os.makedirs(output_dir_checkpoint, exist_ok=True)
                    if torch.cuda.device_count() > 1:
                        model.module.save_pretrained(output_dir_checkpoint)
                    else:
                        model.save_pretrained(output_dir_checkpoint)
                    tokenizer.save_pretrained(output_dir_checkpoint)
                    print(
                        f"Saving model checkpoint to {output_dir_checkpoint}"
                    )

        print(
            f"Epoch {epoch} Loss {epoch_loss / len(train_dataloader):.4f} Time {(time.time() - epoch_start_time):.2f}s"
        )

        # Save model at end of each epoch
        output_dir_epoch = os.path.join(output_dir, f"epoch-{epoch}")
        os.makedirs(output_dir_epoch, exist_ok=True)
        if torch.cuda.device_count() > 1:
            model.module.save_pretrained(output_dir_epoch)
        else:
            model.save_pretrained(output_dir_epoch)
        tokenizer.save_pretrained(output_dir_epoch)


if __name__ == "__main__":
    main()
