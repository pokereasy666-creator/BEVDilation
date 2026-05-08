# Copyright (c) OpenMMLab. All rights reserved.
"""Optimizer hook that skips a step when the loss / gradients are non-finite.

Background. With fp32 training there is no dynamic loss scaler to bail us out
of a single bad batch. clip_grad_norm_ does not sanitize nan / inf — it just
propagates them, and the next optimizer.step() writes nan into the weights
and the model is unrecoverable. This hook checks the loss before backward
and the gradients after backward; if either is non-finite, we drop that
accumulation chunk on the floor, log a warning, and continue.

Instrumentation: tracks skip counts and a sliding 1000-entry window of
pre-clip gradient L2 norms (the value clip_grad_norm_ returns). Every
_report_every iters emits one INFO line summarising skip rate and norm
distribution. The pre-clip capture assumes grad_clip is configured; with
grad_clip=None the parent never calls clip_grads and the window stays empty.
"""
from collections import deque

import numpy as np
import torch
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.optimizer import GradientCumulativeOptimizerHook


@HOOKS.register_module()
class NanSkipGradientCumulativeOptimizerHook(GradientCumulativeOptimizerHook):

    _report_every = 1000

    def _init_instrumentation(self):
        self._total_iters = 0
        self._loss_skipped = 0
        self._grad_skipped = 0
        self._pre_clip_norms = deque(maxlen=1000)

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)
        if not hasattr(self, '_total_iters'):
            self._init_instrumentation()
        self._total_iters += 1

        loss = runner.outputs['loss']
        if not torch.isfinite(loss):
            self._loss_skipped += 1
            runner.logger.warning(
                f'iter {runner.iter}: non-finite loss {loss.item()!r}; '
                f'dropping this accumulation chunk and zeroing grads')
            runner.optimizer.zero_grad()
            self._maybe_report(runner)
            return

        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = loss / loss_factor
        loss.backward()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):
            params = [p for p in runner.model.parameters()
                      if p.requires_grad and p.grad is not None]
            grads_finite = all(torch.isfinite(p.grad).all() for p in params)
            if not grads_finite:
                self._grad_skipped += 1
                runner.logger.warning(
                    f'iter {runner.iter}: non-finite grads after backward; '
                    f'dropping this optimizer step and zeroing grads')
                runner.optimizer.zero_grad()
                self._maybe_report(runner)
                return

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    self._pre_clip_norms.append(float(grad_norm))
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)},
                        runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()

        self._maybe_report(runner)

    def _maybe_report(self, runner):
        if self._total_iters % self._report_every != 0:
            return
        total = self._total_iters
        loss_skip = self._loss_skipped
        grad_skip = self._grad_skipped
        skip_rate = 100.0 * (loss_skip + grad_skip) / max(total, 1)
        snapshot = list(self._pre_clip_norms)
        if snapshot:
            arr = np.asarray(snapshot)
            stats = (
                f'pre_clip_norm[min/p50/p95/p99/max]='
                f'{float(arr.min()):.4g}/'
                f'{float(np.percentile(arr, 50)):.4g}/'
                f'{float(np.percentile(arr, 95)):.4g}/'
                f'{float(np.percentile(arr, 99)):.4g}/'
                f'{float(arr.max()):.4g}')
        else:
            stats = 'pre_clip_norm[min/p50/p95/p99/max]=n/a (no optimizer steps yet)'
        runner.logger.info(
            f'[nan-skip] iter={runner.iter} total={total} '
            f'loss_skip={loss_skip} grad_skip={grad_skip} '
            f'skip_rate={skip_rate:.2f}% {stats} (window={len(snapshot)})')
