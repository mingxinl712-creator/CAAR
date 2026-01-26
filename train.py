import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from Evo1_post import EVO1
from accelerate import Accelerator 
import logging
from accelerate import Accelerator
from torch.optim import AdamW

accelerator = Accelerator(mixed_precision="bf16")

## 记录模型训练参数数量
def inspect_named_submodules(module_dict: dict, verbose: bool = True):
    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)

## 定义学习率曲线
def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda

## 初始化logging
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")

## 训练步骤日志记录
def log_training_step(step, loss, scheduler, dataloader, accelerator):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss:.4f}")
        logging.info(f"learning_rate: {scheduler.get_last_lr()[0]:.5f}")

## 保存检查点
def save_checkpoint(model, checkpoint_dir, cfg, step, loss):
    if accelerator is None or accelerator.is_main_process:
        checkpoint = {
            "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
            "args": cfg,
        }
        checkpoint_path = f"{checkpoint_dir}/{step:07d}_{loss:.6f}.pt"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved best checkpoint at step {step} with loss {loss:.6f}")

## 检查数值稳定性
def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

## 构建参数组以应用不同的权重衰减
def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

@torch.no_grad()
def evaluate_sampling_loss(
    model,
    dataset,
    accelerator,
    batch_size: int,
    num_iter: int = 4,
):
    model.eval()
    running_loss = 0.0
    n_batches = 0

    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    eval_iter = iter(eval_loader)

    for _ in range(num_iter):
        try:
            batch = next(eval_iter)
        except StopIteration:
            eval_iter = iter(eval_loader)
            batch = next(eval_iter)

        prompts = batch["prompts"]
        images_batch = batch["images"]        # (B, V, 3, 448, 448)
        image_masks = batch["image_mask"]     # (B, V)

        # ⭐ 不要 cast 成 bf16，直接保持 float32
        states = batch["states"].to(device=accelerator.device, dtype=torch.float32)
        actions_gt = batch["actions"].to(device=accelerator.device, dtype=torch.float32)

        fused_tokens_list = []
        fused_mask_list = []

        backbone = model.module if hasattr(model, "module") else model

        for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
            fused, fused_mask = backbone.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=prompt,
            )
            fused_tokens_list.append(fused.to(device=accelerator.device, dtype=torch.float32))
            fused_mask_list.append(fused_mask.to(device=accelerator.device, dtype=torch.float32))

        fused_tokens = torch.cat(fused_tokens_list, dim=0)   # (B, S, D)
        fused_mask = torch.cat(fused_mask_list, dim=0)       # (B, S)

        # 采样动作：不用 bf16，直接 float32 就行
        sampled_actions = backbone.predict_actions(
            fused_tokens=fused_tokens,
            fused_mask=fused_mask,
            state=states,
            actions_gt=None,
        )  # (B, seq_len, action_dim)

        # MSE 也用 float32 就好
        mse = (sampled_actions.float() - actions_gt.float()).pow(2).mean()
        mse = accelerator.reduce(mse, reduction="mean")

        running_loss += mse.item()
        n_batches += 1

    model.train()
    return running_loss / max(1, n_batches)

def train(cfg):
    # === Set logging ===
    setup_logging(log_dir=cfg.save_dir)

    # === Dataset ===
    from dataset.Dataset import MetaWorld_Dataset 
    dataset = MetaWorld_Dataset(args=cfg.dataset_args)

    # === Dataset ===
    #from dataset.Dataset import Libero_Dataset 
    #dataset = Libero_Dataset(args=cfg.dataset_args)

    # === DataLoader ===
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )

    # === Model ===
    cfg.device = str(accelerator.device)
    model = EVO1(cfg)
    model.set_finetune_flags()

    lr = cfg.lr
    wd = cfg.weight_decay
    optimizer = AdamW(build_param_groups(model, wd), lr=lr)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")
    
    # === Warmup + Cosine Scheduler ===
    max_steps = cfg.max_steps
    warmup_steps = cfg.warmup_steps
    save_dir = cfg.save_dir

    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = cfg.log_interval
    ckpt_interval = cfg.ckpt_interval

    # === Resume training from checkpoint ===
    resume = cfg.resume
    resume_path = cfg.resume_path

    if resume:
        checkpoint = torch.load(resume_path, map_location=accelerator.device)
        model.load_state_dict(checkpoint["model"])
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_path}")
    else:
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    # 自定义学习率曲线
    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps))

    if accelerator.is_main_process:
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_expert": model.action_expert
        })

    # === Training Loop ===
    model.train()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    step = 0
    running_loss = 0
    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_mask"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            fused_tokens_list = []
            fused_mask_list = []

            for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                fused, fused_mask = model.module.get_vl_embeddings(images=images, image_mask=image_mask, prompt=prompt)
                fused_tokens_list.append(fused.to(dtype=torch.bfloat16))
                fused_mask_list.append(fused_mask.to(dtype=torch.bfloat16))

            fused_tokens = torch.cat(fused_tokens_list, dim=0) # shape (B, S, D)
            fused_mask = torch.cat(fused_mask_list, dim=0)   # shape (B, S)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity, noise, m = model(fused_tokens, fused_mask, state=states, actions_gt=actions_gt)

            #target_velocity = (actions_gt - noise)
            target_velocity = actions_gt.unsqueeze(2).expand_as(pred_velocity) - noise
            assert pred_velocity.shape == target_velocity.shape
            import torch.nn.functional as F
            mse = F.mse_loss(pred_velocity, target_velocity, reduction="none")
            loss = (mse * m).sum() / (m.sum() * mse.shape[-1] + 1e-8)

            # === NaN/Inf check ===
            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                loss=loss
            ):
                continue

            # === Backward and optimizer step ===
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            loss_detached = loss.detach().float()
            loss_mean = accelerator.reduce(loss_detached, reduction="mean")
            running_loss += loss_mean
            step += 1
            # === Logging ===
            if step % log_interval == 0 or step == 1:
                avg_loss = running_loss / log_interval
                avg_loss = avg_loss.item()
                log_training_step(step, avg_loss, scheduler, dataloader, accelerator)
                running_loss = 0.0
                # === 只在主进程打印 sampling loss，所有进程都要参与计算 ===
                sampling_loss = evaluate_sampling_loss(
                    model=model,
                    dataset=dataset,                 # 传 dataset，不是 dataloader
                    accelerator=accelerator,
                    batch_size=cfg.batch_size,
                    num_iter=getattr(cfg, "eval_num_iter", 4),
                )
                if accelerator.is_main_process:
                    logging.info(f"[SamplingEval] step={step}, sampling_mse={sampling_loss:.6f}")
                # === Save best checkpoint ===
                if avg_loss < best_loss and step > warmup_steps:
                    best_loss = avg_loss
                    save_checkpoint(model, save_dir, cfg, step=step, loss=avg_loss)
            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                save_checkpoint(model, save_dir, cfg, step=step, loss=avg_loss)

    # === Save final model ===
    save_checkpoint(model, save_dir, cfg, step=step, loss=avg_loss)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("/home/liumingxin/learning/ICML2026_1/Evo-1/Evo_1/config/config_post_metaworld.yaml")
    try:
        train(cfg)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)

