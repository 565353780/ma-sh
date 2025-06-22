class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        :param patience: 允许的最大容忍次数，即当 metric 不再改善时，最多可以容忍多少个 epoch。
        :param min_delta: 最小的改善值，只有当 metric 改变大于该值时才视为有效的改善。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, metric):
        # 初始时，保存第一个 metric 作为最佳 metric
        if self.best_metric is None:
            self.best_metric = metric
            return False

        # 如果当前 metric 没有明显改善，则增加计数器
        if metric > self.best_metric - self.min_delta:
            self.counter += 1
        else:
            self.best_metric = metric
            self.counter = 0  # 如果有改善，重置计数器

        # 如果连续的 patience 次没有改善，停止训练
        if self.counter >= self.patience:
            self.stop_training = True

        return self.stop_training
