import sys
import time

import numpy as np
from pytorch_lightning.callbacks import ProgressBarBase


class Kbar(object):
    """Keras progress bar. Taken from the pkbar python package.
    Credits goes to author.
    Arguments:
            target: Total number of steps expected, None if unknown.
            width: Progress bar width on screen.
            verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
            stateless_metrics: Iterable of string names of metrics that
                    should be averaged over time. Metrics in this list
                    will be will be averaged by the progbar before display.
                    All others will be displayed as-is.
            interval: Minimum visual progress update interval (in seconds).
            unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
        self,
        target,
        width=30,
        verbose=1,
        interval=0.05,
        stateless_metrics=None,
        unit_name="step",
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateless_metrics:
            self.stateless_metrics = set(stateless_metrics)
        else:
            self.stateless_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
                current: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateless_metrics`,
                        `value_for_last_step`, an average of the metric over time will be displayed.
                        Else, the metric will be displayed as-is.
        """
        values = values or []
        for k, v in values:
            # if torch tensor, convert it to numpy
            if str(type(v)) == "<class 'torch.Tensor'>":
                v = v.detach().cpu().numpy()

            if k not in self._values_order:
                self._values_order.append(k)
            if k in self.stateless_metrics:
                if k not in self._values:
                    self._values[k] = [
                        v * (current - self._seen_so_far),
                        current - self._seen_so_far,
                    ]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += current - self._seen_so_far
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = " - %.0fs" % (now - self._start)
        if self.verbose == 1:
            if (
                now - self._last_update < self.interval
                and self.target is not None
                and current < self.target
            ):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write("\b" * prev_total_width)
                sys.stdout.write("\r")
            else:
                sys.stdout.write("\n")

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = " - ETA: %s" % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += " %.0fs/%s" % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += " %.0fms/%s" % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += " %.0fus/%s" % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += " - %s:" % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                else:
                    info += " %s" % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if self.target is not None and current >= self.target:
                info += "\n"

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += " - %s:" % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                info += "\n"

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class ProgbarCallback(ProgressBarBase):
    def __init__(self):
        """
        Progress bar that uses an implementation of `pkbar<https://pypi.org/project/pkbar/>`_.
        """
        super(ProgbarCallback, self).__init__()

        self.enabled = True
        self.train_prog_bar = None

        self.train_metrics = []
        self.val_metrics = []

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def init_train_bar(self):
        """
        Initialized the progress bar to be used for training.

        Returns:
            Kbar: The progress bar.
        """

        return Kbar(self.total_train_batches)

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)

        if self.enabled:
            print(
                "Epoch: %d / %d" % (trainer.current_epoch + 1, trainer.max_epochs),
                flush=True,
            )
            self.train_prog_bar = self.init_train_bar()

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, batch, batch_idx, dataloader_idx)

        if len(self.train_metrics) == 0:
            self._populate_train_metrics(trainer.progress_bar_dict)

        if self.enabled:
            values = [
                (name, np.array(trainer.progress_bar_dict[name]).astype(np.float32))
                for name in self.train_metrics
            ]

            self.train_prog_bar.update(batch_idx, values=values)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)

        if self.enabled:
            self.train_prog_bar.add(1)

            # printing validation stats here since on_validation_epoch_end
            # is called before on_train_epoch_end.
            if len(self.val_metrics) == 0:
                self._populate_val_metrics(trainer.progress_bar_dict)

            val_str = []
            for metric in self.val_metrics:
                val_str.append(
                    "{}: {:.4f}".format(metric, trainer.progress_bar_dict[metric])
                )
            print(f"Validation set: {', '.join(val_str)}\n", flush=True)

    def _populate_train_metrics(self, progress_bar_dict):
        for key in progress_bar_dict:
            if key != "v_num":
                self.train_metrics.append(key)

    def _populate_val_metrics(self, progress_bar_dict):
        for key in progress_bar_dict:
            if key != "v_num" and key != "loss" and key not in self.train_metrics:
                self.val_metrics.append(key)
