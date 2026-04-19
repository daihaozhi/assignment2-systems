from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.optim as optim


class ShardedOptimizer(optim.Optimizer):
    """
    A simple optimizer-state sharding wrapper.

    - Each parameter is assigned to one owner rank (round-robin by parameter order).
    - Only owner rank updates that parameter and keeps optimizer state for it.
    - After local update, owner broadcasts the updated parameter to all ranks.
    """

    def __init__(
        self,
        params,
        optimizer_cls: type[optim.Optimizer],
        **kwargs: Any,
    ):
        if optimizer_cls is None:
            raise ValueError("optimizer_cls must not be None.")

        self.optimizer_cls = optimizer_cls
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # Let Optimizer normalize input params/param_groups.
        super().__init__(params, kwargs)

        # Global deterministic ownership metadata.
        self._param_owner_by_id: dict[int, int] = {}
        self._ordered_unique_params: list[torch.nn.Parameter] = []
        self._next_param_index = 0

        # Local wrapped optimizer (only this rank's shard).
        self._local_optimizer: optim.Optimizer | None = None
        self._initialize_local_optimizer()

    def _build_local_param_group(self, group: dict[str, Any], assign_new: bool) -> dict[str, Any] | None:
        local_params = []
        for p in group["params"]:
            pid = id(p)
            if pid not in self._param_owner_by_id:
                if not assign_new:
                    continue
                owner = self._next_param_index % self.world_size
                self._param_owner_by_id[pid] = owner
                self._ordered_unique_params.append(p)
                self._next_param_index += 1
            if self._param_owner_by_id[pid] == self.rank:
                local_params.append(p)

        if not local_params:
            return None

        local_group = {k: v for k, v in group.items() if k != "params"}
        local_group["params"] = local_params
        return local_group

    def _initialize_local_optimizer(self) -> None:
        local_groups = []
        for group in self.param_groups:
            local_group = self._build_local_param_group(group, assign_new=True)
            if local_group is not None:
                local_groups.append(local_group)

        if local_groups:
            self._local_optimizer = self.optimizer_cls(local_groups)
        else:
            self._local_optimizer = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        local_loss = None
        if self._local_optimizer is not None:
            local_loss = self._local_optimizer.step()

        # Synchronize updated parameters from owner ranks.
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            for p in self._ordered_unique_params:
                owner = self._param_owner_by_id[id(p)]
                dist.broadcast(p.data, src=owner)

        return loss if loss is not None else local_loss

    def add_param_group(self, param_group: dict[str, Any]):
        # Add to global optimizer metadata first.
        super().add_param_group(param_group)

        # The canonicalized group is appended by Optimizer.add_param_group.
        canonical_group = self.param_groups[-1]
        local_group = self._build_local_param_group(canonical_group, assign_new=True)
        if local_group is None:
            return

        if self._local_optimizer is None:
            self._local_optimizer = self.optimizer_cls([local_group])
        else:
            self._local_optimizer.add_param_group(local_group)
