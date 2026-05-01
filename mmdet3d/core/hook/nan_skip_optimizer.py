# Copyright (c) OpenMMLab. All rights reserved.
"""Optimizer hook that skips a step when the loss / gradients are non-finite.

Background. With fp32 training there is no dynamic loss scaler to bail us out
of a single bad batch. clip_grad_norm_ does not sanitize nan / inf — it just
propagates them, and the next optimizer.step() writes nan into the weights
and the model is unrecoverable. This hook checks the loss before backward
and the gradients after backward; if either is non-finite, we drop that
accumulation chunk on the floor, log a warning, and continue.
"""
import torch
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.optimizer import GradientCumulativeOptimizerHook


@HOOKS.register_module()
class NanSkipGradientCumulativeOptimizerHook(GradientCumulativeOptimizerHook):

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        loss = runner.outputs['loss']
        if not torch.isfinite(loss):
            runner.logger.warning(
                f'iter {runner.iter}: non-finite loss {loss.item()!r}; '
                f'dropping this accumulation chunk and zeroing grads')
            runner.optimizer.zero_grad()
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
                runner.logger.warning(
                    f'iter {runner.iter}: non-finite grads after backward; '
                    f'dropping this optimizer step and zeroing grads')
                runner.optimizer.zero_grad()
                return

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)},
                        runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()
