"""Shared mutable app state — populated in main.lifespan, read by routers."""

import torch

resources: dict = {}
# Ephemeral in-memory store for user-computed prototypes keyed by session_id
session_store: dict[str, dict[int, torch.Tensor]] = {}
