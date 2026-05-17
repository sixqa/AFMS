from mmcv.transforms import Compose
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ConditionalPipelineSwitchHook(Hook):
    """当验证指标连续若干次没有提升时，切换到第二阶段 pipeline。

    注意：
    - patience 计数单位是 validation 次数，不是 epoch。
    - 如果 val_interval=5，patience=8 等价于约 40 个 epoch。
    """

    def __init__(
        self,
        monitor='PCK',
        rule='greater',
        patience=10,
        min_delta=0.001,
        switch_pipeline=None,
        strict=True,
    ):
        self.monitor = monitor
        self.rule = rule
        self.patience = patience
        self.min_delta = min_delta
        self.switch_pipeline = switch_pipeline or []
        self.strict = strict

        self.best_score = None
        self.bad_count = 0
        self._has_switched = False
        self._restart_dataloader = False

        if self.rule not in ['greater', 'less']:
            raise ValueError("rule must be 'greater' or 'less'")

    def _is_improved(self, current):
        if self.best_score is None:
            return True

        if self.rule == 'greater':
            return current > self.best_score + self.min_delta
        else:
            return current < self.best_score - self.min_delta

    def before_train_epoch(self, runner):
        if self._restart_dataloader:
            runner.train_dataloader._DataLoader__initialized = True
            self._restart_dataloader = False

    def after_val_epoch(self, runner, metrics):
        if self._has_switched:
            return

        if self.monitor not in metrics:
            msg = (
                f'{self.monitor} not found in validation metrics. '
                f'Available metrics: {list(metrics.keys())}'
            )
            if self.strict:
                raise KeyError(msg)
            else:
                runner.logger.warning(msg)
                return

        current = metrics[self.monitor]

        if self._is_improved(current):
            self.best_score = current
            self.bad_count = 0
        else:
            self.bad_count += 1

        runner.logger.info(
            f'[ConditionalPipelineSwitchHook] '
            f'{self.monitor}={current:.6f}, '
            f'best={self.best_score:.6f}, '
            f'bad_count={self.bad_count}/{self.patience}'
        )

        if self.bad_count >= self.patience:
            runner.logger.info(
                f'Switch pipeline now: {self.monitor} has not improved '
                f'for {self.bad_count} validation checks.'
            )

            train_loader = runner.train_dataloader
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)

            if getattr(train_loader, 'persistent_workers', False):
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True

            self._has_switched = True