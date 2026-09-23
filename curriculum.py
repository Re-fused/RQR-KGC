"""Curriculum helpers for relation-pool hard negatives."""

import numpy as np
import torch


def adaptive_hard_count_cap(pool_size, nentity, negative_size,
                            min_ratio, max_ratio, available):
    ratio = np.clip(max_ratio * (1.0 - pool_size / float(nentity)),
                    min_ratio, max_ratio)
    return min(int(round(negative_size * ratio)), available)


def curriculum_progress(step, start, ramp):
    if step <= start:
        return 0.0
    return min(1.0, (step - start) / float(max(ramp, 1)))


def select_curriculum_negatives(uniform, hard, cap, progress):
    count = torch.round(cap.float() * progress).long()
    count = count.clamp(min=0, max=min(uniform.size(1), hard.size(1)))
    positions = torch.arange(uniform.size(1)).unsqueeze(0)
    start = uniform.size(1) - count.unsqueeze(1)
    mask = positions >= start
    indices = (positions - start).clamp(min=0, max=max(hard.size(1) - 1, 0))
    return torch.where(mask, torch.gather(hard, 1, indices), uniform)
