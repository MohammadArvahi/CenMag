"""Disk capacity for local checkpoint publication."""
import os
import shutil


def available_capacity(root, reserve, estimated_checkpoint_bytes):
    used = 0
    seen = set()
    for folder, _, names in os.walk(root, followlinks=False):
        for name in names:
            try:
                stat = os.stat(os.path.join(folder, name), follow_symlinks=False)
            except FileNotFoundError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                used += stat.st_blocks * 512
                seen.add(key)
    free = shutil.disk_usage(root).free
    needed = estimated_checkpoint_bytes * 8 + reserve
    return dict(used_bytes=used, free_bytes=free, reserve_bytes=needed, budget_bytes=None, safe=free >= needed)
