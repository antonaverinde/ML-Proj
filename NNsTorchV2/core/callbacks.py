import gc
import os

import psutil
import torch


class Callback:
    """Base callback class for PyTorch training loops."""

    def on_epoch_end(self, epoch, model, optimizer, logs=None):
        pass


class SafeModelCheckpoint(Callback):
    """Save model checkpoint and verify the file is valid."""

    def __init__(self, filepath, save_best_only=False, monitor='val_loss', mode='min'):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, model, optimizer, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                return
            if (self.mode == 'min' and current >= self.best) or \
               (self.mode == 'max' and current <= self.best):
                return
            self.best = current

        #torch.save(model.state_dict(), filepath)
        #full resume
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, filepath)

        try:
            torch.load(filepath, weights_only=True)
        except Exception as e:
            print(f"[WARNING] Failed to verify checkpoint: {filepath}\n{e}")


class DebugOpenFilesCallback(Callback):
    """Check for open .pt/.pth files at epoch end."""

    def on_epoch_end(self, epoch, model, optimizer, logs=None):
        proc = psutil.Process(os.getpid())
        open_files = proc.open_files()
        pt_files = [f.path for f in open_files if f.path.endswith(('.pt', '.pth', '.h5'))]

        if pt_files:
            print(f"[DEBUG] Open model files after epoch {epoch}:\n" + "\n".join(pt_files))


class MemoryCleanupCallback(Callback):
    """Perform garbage collection after each epoch."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def on_epoch_end(self, epoch, model, optimizer, logs=None):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"[Memory] After epoch {epoch}: RSS={mem_info.rss / 1024**2:.1f}MB")


class LearningRateLogger(Callback):
    """Log current learning rate."""

    def __init__(self, writer=None):
        self.writer = writer  # TensorBoard SummaryWriter

    def on_epoch_end(self, epoch, model, optimizer, logs=None):
        logs = logs if logs is not None else {}
        lr = optimizer.param_groups[0]['lr']
        logs['learning_rate'] = lr

        if self.writer:
            self.writer.add_scalar('learning_rate', lr, epoch)
