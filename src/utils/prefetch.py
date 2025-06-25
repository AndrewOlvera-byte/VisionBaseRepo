from typing import Iterable, Iterator

import torch


class PrefetchLoader(Iterable):
    """Wrap a DataLoader to asynchronously preload the next batch onto GPU.

    This reduces GPU idle time by copying input tensors to CUDA stream while 
    the current batch is executing.
    """

    def __init__(self, loader, amp: bool = True):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.amp = amp
        self.next_data = None
        self._preload()

    def _cuda(self, batch):
        if isinstance(batch, (list, tuple)):
            converted = [self._cuda(x) for x in batch]
            return tuple(converted) if isinstance(batch, tuple) else converted
        elif isinstance(batch, dict):
            return {k: self._cuda(v) for k, v in batch.items()}
        elif torch.is_tensor(batch):
            return batch.cuda(non_blocking=True)
        else:
            return batch

    def _preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            self.next_data = self._cuda(self.next_data)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_data
        self._preload()
        return batch 