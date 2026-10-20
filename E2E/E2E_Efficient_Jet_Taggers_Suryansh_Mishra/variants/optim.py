"""Optimizers and parameter averaging for the ablation study: Lion, Lookahead, EMA.

Lifted from the project's earlier hybrid-transformer tree
(``src/optim/{lion,lookahead,ema}.py``), consolidated into a single module so
the optimizer ablation arms are self-contained (DD10).

References:
    Lion: Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023.
        https://arxiv.org/abs/2302.06675
    Lookahead: Zhang, Lucas, Ba, Hinton, "Lookahead Optimizer: k steps
        forward, 1 step back", NeurIPS 2019. https://arxiv.org/abs/1907.08610
    EMA: Polyak averaging — Polyak & Juditsky, "Acceleration of Stochastic
        Approximation by Averaging", 1992.
"""

from collections import defaultdict
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

__all__ = ["Lion", "Lookahead", "EMA"]


class Lion(Optimizer):
    """
    EvoLved Sign Momentum (Lion) Optimizer.

    Reference:
        Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023.
        https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update = sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1).sign_()

                # Step
                p.add_(update, alpha=-group['lr'])

                # Update momentum: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class Lookahead(Optimizer):
    """
    PyTorch implementation of the Lookahead optimizer wrapper.

    Parameters
    ----------
    optimizer: Optimizer
        The inner optimizer.
    la_steps: int
        Number of lookahead steps.
    la_alpha: float
        Linear interpolation factor. 1.0 recovers the inner optimizer.
    pullback_momentum: str
        Change to inner optimizer momentum on interpolation update.

    .. References::
        Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E Hinton.
        [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610).
        *Advances in Neural Information Processing Systems*, 32, 2019.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        la_steps: int = 5,
        la_alpha: float = 0.8,
        pullback_momentum: str = 'none'
    ):
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self, set_to_none: bool = True):
        """Clear inner-optimizer gradients.

        ``set_to_none`` is forwarded so ``Lookahead`` stays interchangeable with a
        plain ``torch.optim.Optimizer`` at the call site (PyTorch made
        ``set_to_none=True`` the default in 2.0, and it avoids keeping a
        zero-filled gradient buffer per parameter).
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        """Serialize the inner optimizer plus Lookahead's slow-weight cache.

        The parent ``Optimizer.state_dict`` contract is only half the story here:
        Lookahead also holds, per parameter, the *slow* weights (``cached_params``
        and optionally ``cached_mom``) that ``step()`` interpolates toward every
        ``la_steps`` iterations.  Discarding them on save means a resumed run
        blends trained weights back toward whatever the optimizer was constructed
        on (fresh random init), which silently corrupts training.  The cache is
        keyed by flat parameter index rather than by tensor so it survives a
        process restart.
        """
        inner = self.optimizer.state_dict()

        params = [p for group in self.optimizer.param_groups for p in group['params']]
        index = {id(p): i for i, p in enumerate(params)}
        cache = {}
        for p, param_state in self.state.items():
            if id(p) not in index:
                continue
            entry = {'cached_params': param_state['cached_params'].cpu().clone()}
            if 'cached_mom' in param_state:
                entry['cached_mom'] = param_state['cached_mom'].cpu().clone()
            cache[str(index[id(p)])] = entry

        return {
            'inner': inner,
            'cache': cache,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum,
        }

    def load_state_dict(self, state_dict):
        """Restore the inner optimizer and the Lookahead slow-weight cache.

        Accepts both the current nested format and the legacy format (a bare
        inner-optimizer ``state_dict``).  For a legacy checkpoint there is no
        cache to restore: ``cached_params`` is re-primed from the current
        parameter values, which by the time this runs are the checkpoint model
        weights (``train.py`` loads the model before the optimizer).  The first
        sync is then a no-op instead of blending toward random init.
        """
        inner = state_dict.get('inner', state_dict)
        self.optimizer.load_state_dict(inner)

        params = [p for group in self.optimizer.param_groups for p in group['params']]
        cache = state_dict.get('cache')
        if cache is not None:
            for i, p in enumerate(params):
                entry = cache.get(str(i))
                if entry is None:
                    continue
                param_state = self.state[p]
                param_state['cached_params'].copy_(
                    entry['cached_params'].to(p.device)
                )
                if 'cached_mom' in entry and 'cached_mom' in param_state:
                    param_state['cached_mom'].copy_(
                        entry['cached_mom'].to(p.device)
                    )
            self._la_step = int(state_dict.get('_la_step', 0))
        else:
            # Legacy checkpoint: prime the slow-weight cache from the current
            # (already checkpoint-loaded) parameters and restart the sync clock.
            for p in params:
                param_state = self.state[p]
                param_state['cached_params'].copy_(p.data)
                if 'cached_mom' in param_state:
                    # A momentum BUFFER, not weights.  ``pullback_momentum`` blends
                    # this into the inner optimizer's momentum at the next sync, so
                    # seeding it from ``p.data`` (as the line above legitimately
                    # does for cached_params) would inject weight magnitudes into
                    # the momentum state.  The inner state_dict just loaded carries
                    # the real buffer; zero is the correct fallback when the inner
                    # optimizer has not built one yet.
                    inner = self.optimizer.state.get(p, {})
                    momentum = inner.get('momentum_buffer')
                    if momentum is not None:
                        param_state['cached_mom'].copy_(momentum)
                    else:
                        param_state['cached_mom'].zero_()
            self._la_step = 0

    def _backup_and_load_cache(self):
        """
        Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])

                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """
        Performs a single Lookahead optimization step.

        Parameters
        ----------
            closure: Callable, optional
                A closure that reevaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1
        if self._la_step >= self._total_la_steps:
            self._la_step = 0

            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"]
                        )
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters updated as:
        ema_param = decay * ema_param + (1 - decay) * model_param

    During evaluation, swap in EMA weights for smoother, more stable predictions.

    Reference:
        Polyak averaging — Polyak & Juditsky, "Acceleration of Stochastic
        Approximation by Averaging", 1992.
        Commonly used in modern training: diffusion models, vision transformers, etc.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters to track.
    decay : float
        EMA decay factor. Higher = smoother (0.999 typical for long training,
        0.99 for short). Default 0.999.
    warmup_steps : int
        Number of steps before EMA starts. During warmup, decay is ramped
        linearly from 0 to `decay` to avoid copying random init weights.
        Default 0 (no warmup).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Shadow parameters (deep copy of model params, detached)
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        # Backup storage for swap
        self._backup = {}

    def _get_decay(self) -> float:
        """Ramp decay from 0 to self.decay during warmup."""
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, self.step_count / self.warmup_steps * self.decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters. Call after optimizer.step()."""
        self.step_count += 1
        d = self._get_decay()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def apply_shadow(self, model: nn.Module):
        """Swap model params with EMA params (for evaluation)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params (after evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'step_count': self.step_count,
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict['warmup_steps']
        for k, v in state_dict['shadow'].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
