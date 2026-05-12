import argparse
import json
import os
import random
import re
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
import wandb
from loguru import logger
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM as HF_LlamaForCausalLM,
)

import datasets
import datasets.distributed

from safetensors.torch import load_file

from training import args_utils, training_utils
from training.dataloader import BufferDataset, PreprocessedDataset
from training.modeling_llama import LlamaForCausalLM

from metrics.metrics import *

transformers.logging.set_verbosity_error()


def nvbit_on(target_func=None):
    if target_func is not None:
        os.environ['TARGET_FUNC'] = str(target_func)

    os.kill(os.getpid(), signal.SIGUSR1)


def nvbit_off(): 
    os.kill(os.getpid(), signal.SIGUSR2)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=25_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000)
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)    
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--disable_final_evaluation", default=False, action="store_true")
    parser.add_argument("--compare_every", type=int, default=None)
    parser.add_argument("--exit_after", type=int, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--base_model_continue", default=False, action="store_true")
    parser.add_argument("--record_attn_metrics", default=False, action="store_true")
    parser.add_argument("--fi_nvbit_enable", default=False, action="store_true")
    parser.add_argument("--fi_nvbit_location", type=str, default=None)
    parser.add_argument("--fi_nvbit_trigger_rate", type=int, default=None)
    parser.add_argument("--fi_nvbit_recompute", default=False, action="store_true")
    parser.add_argument("--fi_nvbit_alpha", type=float, default=0.02)
    parser.add_argument("--fi_nvbit_target_funcs", nargs="+", type=int, default=None)
    parser.add_argument("--fi_nvbit_duration", type=int, default=1)
    parser.add_argument("--fi_nvbit_duration_random", default=False, action="store_true")
    parser.add_argument("--fi_nvbit_steps", nargs="+", type=int, default=None)

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("allenai/c4", "en", split="validation")
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"]
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    if not args.single_gpu:
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, total_loss)
        total_loss = sum([t.item() for t in gathered_losses]) / world_size
    else:
        total_loss = total_loss.item()
    perplexity = np.exp(total_loss)
    return total_loss, evaluated_on_tokens, perplexity


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.single_gpu:
        assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}, world_size: {world_size}, device_count: {torch.cuda.device_count()}")

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train[:10%]")
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    model_config.pad_token_id = tokenizer.pad_token_id 

    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    last_batch_idx = None
            
    if args.base_model_continue:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.base_model_path}")
        
        checkpoint_path = os.path.join(args.base_model_path, "model.safetensors")
        
        try:
            state_dict = load_file(checkpoint_path)
            #model.load_state_dict(state_dict, strict=True)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model successfully loaded from {checkpoint_path} (strict=True policy)")
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}")
            logger.exception(e)
            raise  # re-raise so you see full traceback if it crashes

        # Load training state (global_step, update_step, etc.)
        training_state_path = os.path.join(args.base_model_path, "training_state.json")
        if os.path.exists(training_state_path):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.base_model_path}")
            with open(training_state_path) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            last_batch_idx = _old_state["batch_idx"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"last_batch_idx: {last_batch_idx}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.base_model_path}, global step will start from zero")

        logger.info("*" * 40)

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        if args.name:
            wandb.init(project=args.project if args.project else None, name=args.name)
        else:
            name=args.save_dir.split("/")[-1]
            wandb.init(project=args.project if args.project else None, name=name)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
        "target_func": os.environ.get("TARGET_FUNC"),
        "target_instr": os.environ.get("TARGET_INSTR"),
        "target_laneid": os.environ.get("TARGET_LANEID"),
        "target_smid": os.environ.get("TARGET_SMID"),
        "target_register": os.environ.get("TARGET_REGISTER"),
        "target_op": os.environ.get("TARGET_OP"),
        "target_bitmask": os.environ.get("TARGET_BITMASK"),
        "target_every": os.environ.get("TARGET_EVERY"),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
        
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9) #args.beta1)       
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio
    )

    # load optimizer and scheduler from checkpoint
    if args.base_model_continue:
        opt_ckpt_path = os.path.join(args.base_model_path, "optimizer.pt")
        if os.path.exists(opt_ckpt_path):
            checkpoint = torch.load(opt_ckpt_path, map_location="cpu", weights_only=False)        
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"Loaded optimizer and scheduler from {opt_ckpt_path}")
        else:
            logger.warning(f"Optimizer checkpoint not found at {opt_ckpt_path}")

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True, 
            static_graph=False 
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    start_batch_idx = last_batch_idx + 1 if last_batch_idx else 0

    if start_batch_idx > 0:
        subset_dataset = torch.utils.data.Subset(dataset, range(start_batch_idx, len(dataset)))
        dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=None, num_workers=args.workers)

    accum_loss = 0.0
    accum_count = 0
    microbatches = []

    trigger_nvbit = False
    nvbit_target_func = None
    nvbit_duration_counter = 0

    # ======= START TRAINING LOOP - we'll never go through all the data, so no need for epochs =======
    for batch_idx, batch in enumerate(dataloader, start=start_batch_idx):

        # ======= CONINTUE FROM CHECKPOINT =======

        if args.base_model_path and args.compare_every and global_step > 0 and global_step % args.compare_every == 0:
            if global_rank == 0:
                try:
                    reference_model_path = os.path.join(args.base_model_path, "model.safetensors")
                    if re.search(r'model_\d+', reference_model_path):
                        reference_model_path = re.sub(r'model_\d+', f'model_{global_step}', reference_model_path)
                    else:
                        dir_path, filename = os.path.split(reference_model_path)
                        reference_model_path = os.path.join(dir_path, f"model_{global_step}", filename)
                        
                    reference_state_dict = load_file(reference_model_path)
                    reference_model = LlamaForCausalLM(model_config)   
                    reference_model.load_state_dict(reference_state_dict, strict=True)

                    if args.dtype in ["bf16", "bfloat16"]:
                        reference_model = reference_model.to(device=device, dtype=torch.bfloat16)
                    else:
                        reference_model = reference_model.to(device=device)

                    parameter_difference = compute_parameter_difference(model, reference_model)
                    wandb.log({"parameter_difference": parameter_difference}, step=global_step,)
                    del reference_model
                except Exception as e:
                    logger.error(e)
                    pass
                    
        if args.exit_after:
            if global_step >= args.exit_after:
                break

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        local_step += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        microbatches.append((batch, labels))

        if (len(microbatches) < args.gradient_accumulation):
            continue

        accum_loss = 0.0
        accum_count = 0

        # ======= FORWARD / BACKWARD =======

        for mb_idx, (mb, labels) in enumerate(microbatches):
            if trigger_nvbit and (args.fi_nvbit_location == "forward" or args.fi_nvbit_location == "forward_backward"):
                nvbit_on(target_func=nvbit_target_func)
                output = model(**mb, labels=labels, use_cache=False, output_attentions=False)
                loss = output.loss
                scaled_loss = loss / args.gradient_accumulation
                torch.cuda.synchronize()
                nvbit_off()
            else:
                output = model(**mb, labels=labels, use_cache=False, output_attentions=False)
                loss = output.loss
                scaled_loss = loss / args.gradient_accumulation

            if trigger_nvbit and (args.fi_nvbit_location == "backward" or args.fi_nvbit_location == "forward_backward"):
                nvbit_on(target_func=nvbit_target_func)
                scaled_loss.backward()
                torch.cuda.synchronize()
                nvbit_off()
            else:
                scaled_loss.backward()

            accum_loss += loss.detach().item()
            accum_count += 1

        # ======= ATTN METRICS =======

        if global_rank == 0 and args.record_attn_metrics:
            attn_entropy_per_layer = []
            max_attn_scores_per_layer = []

            if not args.single_gpu:
                layers = model.module.model.layers
            else:
                layers = model.model.layers
            for layer in layers:
                attn_entropy_per_layer.append(layer.self_attn.compute_attn_scores.entropy)            
                max_attn_scores_per_layer.append(layer.self_attn.compute_attn_scores.max_attn_score)
            attn_entropy = sum(attn_entropy_per_layer) / len(attn_entropy_per_layer)
            max_attn_scores = max(max_attn_scores_per_layer)

        # ======= GRAD BEFORE CLIPPING =======

        gradient_norm_pre = compute_gradient_norm(preprocess(args.single_gpu, model))

        # ======= GRAD CLIPPING =======

        if args.grad_clipping != 0.0: 
            if trigger_nvbit and args.fi_nvbit_location == "clipping":
                nvbit_on()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
                torch.cuda.synchronize()
                nvbit_off()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        # ======= GRAD NORM AFTER CLIPPING =======

        if global_rank == 0:
            gradient_norm_post = compute_gradient_norm(preprocess(args.single_gpu, model))

        # ======= PBAR UPDATE =======

        avg_loss = accum_loss / accum_count
        accum_loss = 0.0
        accum_count = 0

        microbatches.clear()

        if global_rank == 0: 
            pbar.update(1)

        # ======= OPTIMIZER STEP =======

        if trigger_nvbit and args.fi_nvbit_location == "optimizer":
            nvbit_on()
            optimizer.step()
            torch.cuda.synchronize()
            nvbit_off()
        else:
            optimizer.step()

        # ======= NVBIT TRIGGER LOGIC =======

        if global_step > 10:
            if nvbit_duration_counter > 0:
                nvbit_duration_counter -= 1
                trigger_nvbit = True
            else:
                trigger_nvbit = False
                if args.fi_nvbit_enable:
                    if args.fi_nvbit_steps and len(args.fi_nvbit_steps) > 0 and -1 not in args.fi_nvbit_steps:
                        enable_trigger_nvbit = global_step in args.fi_nvbit_steps
                    else:
                        trigger_rate = args.fi_nvbit_trigger_rate
                        enable_trigger_nvbit = random.randint(1, trigger_rate)

                    if enable_trigger_nvbit == 1:
                        if args.fi_nvbit_duration_random:
                            nvbit_duration_counter = random.randint(1, args.fi_nvbit_duration)-1
                        else:
                            nvbit_duration_counter = args.fi_nvbit_duration-1
                        trigger_nvbit = True                    
                        if args.fi_nvbit_target_funcs and len(args.fi_nvbit_target_funcs) > 0 and -1 not in args.fi_nvbit_target_funcs:
                            nvbit_target_func = random.choice(args.fi_nvbit_target_funcs)
                            logger.info(f"[NVBit] Enable FI for the next {nvbit_duration_counter+1} iterations - nvbit_target_func: {nvbit_target_func}") 
                        else:
                            logger.info(f"[NVBit] Enable FI for the next {nvbit_duration_counter+1} iterations") 

        # ======= SCHEDULER and ZERO GRAD =======

        scheduler.step()
        optimizer.zero_grad()

        # ======= UPDATES =======

        global_step += 1

        if global_rank == 0:
            weight_norm = compute_weight_norm(preprocess(args.single_gpu, model))

        update_step += 1
        update_time = time.time() - update_time
        
        # ======= CHECKPOINTING =======

        if args.save_dir and local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            if not args.single_gpu:
                model.module.save_pretrained(current_model_directory, max_shard_size='100GB')
            else:
                model.save_pretrained(current_model_directory, max_shard_size='100GB')

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
                "batch_idx": batch_idx 
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
                
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # ======= EVALUATION =======

        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, _, perplexity = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                        "eval_loss": total_loss,
                        "eval_perplexity": perplexity
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        # ======= LOGGING =======

        lr = optimizer.param_groups[0]["lr"]
        tokens_seen_before = tokens_seen

        if global_rank == 0:
            wandb.log({
                    "loss": avg_loss,
                    "perplexity": np.exp(avg_loss),
                    "lr": lr,
                    "gradient_norm_post": gradient_norm_post,
                    "gradient_norm_pre": gradient_norm_pre,
                    "weight_norm": weight_norm,
                    "update_step": update_step
                },
                step=global_step,
            )

            if args.record_attn_metrics:
                attn_entropy_per_layer_dict = {f"attn_entropy_per_layer/layer_{i}_attn_entropy": score for i, score in enumerate(attn_entropy_per_layer)}
                attn_scores_per_layer_dict = {f"max_attn_logits_per_layer/layer_{i}_max_attn_logits": score for i, score in enumerate(max_attn_scores_per_layer)}
                wandb.log({"attn_entropy": attn_entropy, "max_attention_logits": max_attn_scores}, step=global_step)
                wandb.log(attn_entropy_per_layer_dict, step=global_step)
                wandb.log(attn_scores_per_layer_dict, step=global_step)

        update_time = time.time()

    # ======= END of training loop =======

    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    # ======= CHECKPOINTING =======

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if args.save_dir and global_rank == 0 and not os.path.exists(current_model_directory) and args.save_every > 0:
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        if not args.single_gpu:
            model.module.save_pretrained(current_model_directory)
        else:
            model.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
            "batch_idx": batch_idx
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # ======= FINAL EVALUATION =======
    
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    if not args.disable_final_evaluation:
        total_loss, _, perplexity = evaluate_model(
            model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
        )

        if global_rank == 0:
            wandb.log({
                "eval_loss": total_loss,
                "eval_perplexity": perplexity
                },
                step=global_step,
            )
            logger.info(f"Final eval loss: {total_loss}")

    # ======= FINISH SCRIPT =======

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")
    
    if not args.single_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
