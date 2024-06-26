try:
    from fastchat.train.llama_flash_attn_monkey_patch import (
        replace_llama_attn_with_flash_attn,
    )

    replace_llama_attn_with_flash_attn()
except:
    print('Install fastchat to use flash attention. Refer to https://github.com/lm-sys/FastChat')

import json
from torch.utils.data import Dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AdamW, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from rank_loss import RankLoss
import argparse

class RerankData(Dataset):
    def __init__(self, data, tokenizer, psg_num=20, label=True):
        self.data = data
        self.tokenizer = tokenizer
        self.psg_num = psg_num
        self.label = label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def prompt(query, psg, max_len=200):
        psg = ' '.join(psg.split()[:max_len])
        return f"Predict whether the given passage answer the question.\n\nQuestion: {query}\n\nPassage: {psg}\n\nDoes the passage answer the question?"

    def __getitem__(self, item):
        item = self.data[item]
        query = item['query']

        if self.label:
            pos = [str(item['positive_passages'][0]['text'])]
            pos_id = [psg['docid'] for psg in item['positive_passages']]
            neg = [str(psg['text']) for psg in item['retrieved_passages'] if psg['docid'] not in pos_id][:self.psg_num]
        else:
            pos = []
            neg = [str(psg['text']) for psg in item['retrieved_passages']][:self.psg_num]

        passages = pos + neg
        passages = passages[:self.psg_num]
        passages = passages + ['<padding_passage>'] * (self.psg_num - len(passages))
        data = [self.prompt(query, psg) for psg in passages]
        return data

    def collate_fn(self, data):
        data = sum(data, [])
        batch_size = len(data)
        features = self.tokenizer(data, padding=True, truncation=True, return_tensors="pt",
                                  max_length=2048)

        features['decoder_input_ids'] = torch.zeros(batch_size, 1).long()
        return features


def receive_response(data, responses):
    def clean_response(response: str):
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    new_data = []
    for item, response in zip(data, responses):
        response = clean_response(response)
        response = [int(x) - 1 for x in response.split()]
        response = remove_duplicate(response)
        passages = item['retrieved_passages']
        original_rank = [tt for tt in range(len(passages))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        new_passages = [passages[ii] for ii in response]
        new_data.append({'query': item['query'],
                         'positive_passages': item['positive_passages'],
                         'retrieved_passages': new_passages})
    return new_data


def split_data(data, process_idx, num_processes):
    if isinstance(data, torch.Tensor):
        sublist_length, remainder = divmod(data.size(0), num_processes)
        return data[process_idx * sublist_length + min(process_idx, remainder):(process_idx + 1) * sublist_length + min(
            process_idx + 1, remainder)]
    else:
        return data


def gather_tensors(local_tensor, pad=False):
    if not dist.is_initialized():
        return local_tensor

    if pad:
        local_size = torch.tensor([local_tensor.size(0)], device=local_tensor.device)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)

        max_size = max(torch.stack(sizes)).item()

        padded_tensor = torch.zeros(max_size, *local_tensor.size()[1:], device=local_tensor.device)
        padded_tensor[:local_tensor.size(0)] = local_tensor

        gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, padded_tensor)

        mask = [torch.arange(padded_tensor.size(0), device=padded_tensor.device) < size_tensor.item()
                for size_tensor in sizes]

        gathered_tensors = [gathered_tensor[mask_tensor] for gathered_tensor, mask_tensor in
                            zip(gathered_tensors, mask)]

    else:
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, local_tensor)

    gathered_tensors[dist.get_rank()] = local_tensor

    return torch.cat(gathered_tensors, dim=0)


def train(args):
    model_name = args.model
    loss_type = args.loss
    data_path = args.data
    save_path = args.save_path
    permutation = args.permutation

    accelerator = Accelerator(gradient_accumulation_steps=2)
    batch_size = 16
    psg_num = 8

    # Load model and tokenizer
    if 't5' in model_name:  # flan-t5
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:  # llama
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left", model_max_length=4096)
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    token_yes = tokenizer.get_vocab()[args.token_yes]
    print(f"Token yes: {args.token_yes} = {token_yes}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Load data and permutation
    data = [json.loads(line) for line in open(data_path)]
    response = json.load(open(permutation))
    data = receive_response(data, response)
    dataset = RerankData(data, tokenizer, psg_num=psg_num, label=False)

    # Distributed training
    train_sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                              batch_size=batch_size, num_workers=0,
                                              sampler=train_sampler)

    optimizer = AdamW(model.parameters(), 2e-5)
    model, optimizer, _ = accelerator.prepare(model, optimizer, data_loader)

    loss_function = getattr(RankLoss, loss_type)

    for epoch in range(args.epochs):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            with accelerator.accumulate(model):
                # Split the tensor based on the GPU id
                batch = {k: split_data(v, accelerator.process_index, accelerator.num_processes) for k, v in
                         batch.items()}
                batch = {k: v.cuda() for k, v in batch.items()}

                out = model(**batch)
                logits = gather_tensors(out.logits[:, -1, token_yes].contiguous())  # Gather all predictions across GPUs
                logits = logits.view(-1, psg_num)

                y_true = torch.tensor([[1 / (i + 1) for i in range(logits.size(1))]] * logits.size(0)).cuda()

                loss = loss_function(logits, y_true)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

                loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

        accelerator.wait_for_everyone()
        save_dir = f'{save_path}/{epoch}'
        model.save_checkpoint(save_dir)
        tokenizer.save_pretrained(save_dir)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/flan-t5-xl')
    parser.add_argument('--loss', type=str, default='rank_net')
    parser.add_argument('--data', type=str, default='data/marco-train-10k.jsonl')
    parser.add_argument('--save_path', type=str, default='out/flan-t5-xl-id')
    parser.add_argument('--permutation', type=str, default='marco-train-10k-gpt3.5.json')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument("--token_yes", type=str, default="tak")
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)