import argparse
import json
import math
import os
from contextlib import nullcontext

import torch
from ivon import IVON
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler

from data import get_processed_datasets, get_raw_datasets
from model import WrappedModel, get_model_with_lora
from utils import calculate_metrics, initialize

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def compute_uncertainties(probs: torch.Tensor, eps: float = 1e-12):
    """
    Compute Total, Aleatoric, and Epistemic uncertainty from Monte Carlo logits.

    Args:
        probs (torch.Tensor): Probs of shape (T, B, C), where
            T = number of MC samples,
            B = batch size,
            C = number of classes.
        eps (float): Small constant for numerical stability in log/entropy.

    Returns:
        total_unc (torch.Tensor): Total uncertainty (entropy of mean predictive), shape (B,).
        ale_unc   (torch.Tensor): Aleatoric uncertainty (mean entropy of each predictive), shape (B,).
        epi_unc   (torch.Tensor): Epistemic uncertainty (total_unc - ale_unc), shape (B,).
    """
    # 1. Mean predictive distribution across MC samples
    probs_mean = probs.mean(dim=0)  # (B, C)

    # 2. Total uncertainty: entropy of the mean predictive
    total_unc = -torch.sum(probs_mean * torch.log(probs_mean), dim=1)  # (B,)

    # 3. Aleatoric uncertainty: mean entropy of each predictive distribution
    ale_unc = torch.mean(-torch.sum(probs * torch.log(probs), dim=2), dim=0)  # (B,)

    # 4. Epistemic uncertainty: difference between total and aleatoric
    epi_unc = total_unc - ale_unc  # (B,)

    return total_unc, ale_unc, epi_unc

def parse_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--task_name", type=str, default="winogrande_s", choices=["winogrande_s", "ARC-Challenge", "ARC-Easy", "winogrande_m", "openbookqa", "boolq"])
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--no_tf32", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--tqdm", action="store_true", default=False)

    # Finetuning arguments
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "ivon"])
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--save_to", type=str, default=None)
    parser.add_argument("--json_filename", type=str, default="metrics.json")
    parser.add_argument("--print_freq", type=int, default=50)

    # IVON-specific arguments
    parser.add_argument("--test_num_samples", type=int, default=10)
    parser.add_argument("--ess", type=float, default=1e7)
    parser.add_argument("--hess_init", type=float, default=3e-4)
    parser.add_argument("--clip_radius", type=float, default=1e-3)
    parser.add_argument("--ivon_beta2", type=float, default=0.99999)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.save_to is not None:
        save_path = os.path.join(args.save_to, f"{args.optimizer}_{args.task_name}/{args.seed}")
        os.makedirs(save_path, exist_ok=True)

    initialize(seed=args.seed, deterministic=args.deterministic, tf32=not args.no_tf32)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.bos_token

    model = get_model_with_lora(
        args.model_name_or_path, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    model = WrappedModel(model, args.task_name, tokenizer)
    model.to(device)

    raw_datasets = get_raw_datasets(args.task_name)
    processed_datasets = get_processed_datasets(
        raw_datasets, args.task_name, tokenizer, args.max_length
    )
    train_dataset, eval_dataset = processed_datasets["train"], processed_datasets["validation"]
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    rng = torch.Generator().manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collator, batch_size=args.batch_size, generator=rng
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collator, batch_size=args.batch_size)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.wd,
        )
    elif args.optimizer == "ivon":
        optimizer = IVON(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.wd,
            ess=args.ess,
            hess_init=args.hess_init,
            clip_radius=args.clip_radius,
            beta2=args.ivon_beta2,
            rescale_lr=False,
        )
        logits_list_samples = []

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    completed_steps = 0
    for _ in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        for step, train_batch in enumerate(train_dataloader):
            train_batch = {k: v.to(device) for k, v in train_batch.items()}
            if completed_steps % args.eval_interval == 0:
                model.eval()
                if args.save_to is not None:
                    save_path_step = os.path.join(save_path, f"{completed_steps}")
                    model.model.save_pretrained(save_path_step)

                logits_list = []
                labels_list = []
                with torch.inference_mode():
                    for batch in tqdm(eval_dataloader, disable=not args.tqdm):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        logits_list.append(model(**batch))
                        labels_list.append(batch["labels"])
                        if args.optimizer == "ivon":
                            batch_logits_samples = []
                            for _ in range(args.test_num_samples):
                                with optimizer.sampled_params(train=False):
                                    batch_logits_samples.append(model(**batch))
                            # len(batch_logits_samples) == args.test_num_samples (10)
                            # each batch_logits_samples[i] is logits for one sample: shape (batch_size, num_classes) for winogrande_s (4,2)
                            logits_list_samples.append(batch_logits_samples)

                logits = torch.cat(logits_list, dim=0)
                probs = torch.softmax(logits, dim=-1)
                labels = torch.cat(labels_list, dim=0)
                acc, nll, ece, brier = calculate_metrics(probs, labels)
                print(
                    f"Val: Step {completed_steps} "
                    f"Accuracy: {acc:.4f} NLL: {nll:.4f} ECE: {ece:.4f} Brier: {brier:.4f}"
                )
                if args.save_to is not None:
                    eval_results_path = os.path.join(save_path_step, args.json_filename)
                    json_results = [
                        {"num_samples": "mean", "accuracy": acc, "nll": nll, "ece": ece, "brier": brier}
                    ]

                if args.optimizer == "ivon":
                    probs_sum = torch.zeros_like(logits)
                    all_probs = []
                    for idx in range(args.test_num_samples):
                        logits_sample = torch.cat([batch[idx] for batch in logits_list_samples], dim=0)
                        probs = torch.softmax(logits_sample, dim=-1)
                        probs_sum += probs
                        probs = probs_sum / (idx + 1)
                        all_probs.append(probs)
                        acc, nll, ece, brier = calculate_metrics(probs, labels)

                        print(  
                            f"Val @{idx + 1} samples: Step {completed_steps} "
                            f"Accuracy: {acc:.4f} NLL: {nll:.4f} ECE: {ece:.4f} Brier: {brier:.4f}"
                        )
                        if args.save_to is not None:
                            json_results.append(
                                {"num_samples": idx + 1, "accuracy": acc, "nll": nll, "ece": ece, "brier": brier}
                            )
                    print(f'probs_sum: {probs_sum.shape})')
                    logits_list_samples = []
                    # Compute uncertainties
                    total_unc, ale_unc, epi_unc = compute_uncertainties(torch.stack(all_probs, dim=0))
                    print(  
                            f"TU: {total_unc.mean():.4f} AU: {ale_unc.mean():.4f} EU: {epi_unc.mean():.4f}"
                    )
                if args.save_to is not None:
                    with open(eval_results_path, 'w', newline='', encoding='utf-8') as f:
                        json.dump(json_results, f)

            if completed_steps > args.max_train_steps:
                break

            model.train()
            with optimizer.sampled_params(train=True) if args.optimizer == "ivon" else nullcontext():
                loss = torch.nn.CrossEntropyLoss()(model(**train_batch), train_batch['labels'])
                loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

            if step % args.print_freq == 0:
                print(
                    f"Train: Step {completed_steps:5d} "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.3e} "
                )


if __name__ == "__main__":
    main()
