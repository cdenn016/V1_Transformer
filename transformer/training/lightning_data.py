"""
PyTorch Lightning DataModule for Gauge-Theoretic Transformer
=============================================================

Wraps the existing create_dataloaders() pipeline as a LightningDataModule
for seamless integration with pl.Trainer.

Supports all datasets: wikitext-2, wikitext-103, openwebtext, wiki-ja.

Usage:
    from transformer.training.lightning_data import GaugeDataModule

    dm = GaugeDataModule(
        max_seq_len=256,
        batch_size=64,
        dataset='wikitext-103',
    )
    dm.setup()
    print(f"Vocab size: {dm.vocab_size}")
"""

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl
from torch.utils.data import DataLoader
from typing import Optional

from transformer.data.datasets import create_dataloaders


class GaugeDataModule(pl.LightningDataModule):
    """
    LightningDataModule wrapping the existing data pipeline.

    On setup(), calls create_dataloaders() and stores the resulting
    DataLoaders and vocab_size. The DataLoaders returned by
    train_dataloader() / val_dataloader() are the same objects
    produced by the existing pipeline (preserving caching, worker
    seeding, and tokenizer behavior).
    """

    def __init__(
        self,
        max_seq_len: int = 256,
        batch_size: int = 64,
        vocab_size: Optional[int] = None,
        num_workers: int = 0,
        dataset: str = 'wikitext-103',
        tokenizer_name: str = 'gpt2',
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._vocab_size_arg = vocab_size
        self._num_workers = num_workers
        self._dataset = dataset
        self._tokenizer_name = tokenizer_name
        self._cache_dir = cache_dir

        # Set after setup()
        self.vocab_size: int = 0
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_loader is not None:
            return  # already set up

        train_loader, val_loader, vocab_size = create_dataloaders(
            max_seq_len=self._max_seq_len,
            batch_size=self._batch_size,
            vocab_size=self._vocab_size_arg,
            num_workers=self._num_workers,
            cache_dir=self._cache_dir,
            tokenizer_name=self._tokenizer_name,
            dataset=self._dataset,
        )

        self._train_loader = train_loader
        self._val_loader = val_loader
        self.vocab_size = vocab_size

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader
