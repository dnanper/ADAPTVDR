from transformers import get_cosine_schedule_with_warmup


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, completed_steps: int = 0):
    if completed_steps:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    return get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
        last_epoch=completed_steps - 1,
    )
