"""
Synthetic Gauge Language — Controlled Holonomy for Flat Bundle Experiments.
===========================================================================

Generates (sequence, label) pairs where the label depends on parallel transport
along the sequence through a discrete GL(K) gauge field.

The gauge field assigns each token a local frame g_v in GL+(K) and defines
connections A_{v->w} in gl(K) between token pairs. Holonomy strength epsilon
controls path-dependence:
    epsilon = 0: completely flat — transport depends only on endpoints (path-independent)
    epsilon > 0: non-flat — transport depends on the full path through intermediate tokens

This provides the cleanest falsification test for the flat bundle hypothesis:
    - Flat architecture should excel at epsilon ~ 0 and fail at epsilon >> 0
    - Non-flat architecture should handle all epsilon values
    - Crossover at epsilon* where non-flat surpasses flat
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import expm, logm
from typing import Optional, Tuple


def random_gl_element(K: int, rng: np.random.RandomState, scale: float = 1.0) -> np.ndarray:
    """Sample a random element of GL+(K) near the identity.

    Uses exp(scale * random_matrix) to ensure det > 0 and invertibility.

    Args:
        K: Matrix dimension (gauge group is GL(K)).
        rng: NumPy RandomState for reproducibility.
        scale: Controls distance from identity; larger = more diverse frames.

    Returns:
        (K, K) matrix in GL+(K).
    """
    A = rng.randn(K, K) * scale
    return expm(A)


class SyntheticGaugeLanguage:
    """Discrete gauge field on a token vocabulary with controllable holonomy.

    Each token v ∈ {0, ..., V-1} has a hidden "local frame" g_v ∈ GL(K).
    The connection field A_{v→w} ∈ gl(K) defines transport between tokens:

        Flat part:     A_flat_{v→w} = log(g_v · g_w^{-1})
        Non-flat noise: A_noise_{v→w} ~ N(0, I) in gl(K)
        Total:         A_{v→w} = A_flat + ε · A_noise

    Parallel transport along sequence [t_1, ..., t_n]:
        T = exp(A_{t_1→t_2}) · exp(A_{t_2→t_3}) · ... · exp(A_{t_{n-1}→t_n})

    When ε=0: T = g_{t_1} · g_{t_n}^{-1} (depends only on endpoints!)
    When ε>0: T depends on the full path (intermediate tokens matter)

    Args:
        vocab_size: Number of tokens in the vocabulary.
        K: Dimension of the gauge group GL(K).
        epsilon: Holonomy strength. 0 = flat, >0 = non-flat.
        n_classes: Number of discrete label classes.
        seq_len: Sequence length for generated samples.
        frame_scale: Scale for random frame generation (controls frame diversity).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        K: int = 3,
        epsilon: float = 0.0,
        n_classes: int = 4,
        seq_len: int = 16,
        frame_scale: float = 0.5,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.K = K
        self.epsilon = epsilon
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Fixed local frames g_v ∈ GL+(K) for each token
        self.frames = [random_gl_element(K, rng, scale=frame_scale) for _ in range(vocab_size)]
        self.frames_inv = [np.linalg.inv(g) for g in self.frames]

        # Precompute flat connection: A_flat_{v→w} = log(g_v · g_w^{-1})
        self.A_flat = np.zeros((vocab_size, vocab_size, K, K))
        for v in range(vocab_size):
            for w in range(vocab_size):
                self.A_flat[v, w] = logm(self.frames[v] @ self.frames_inv[w]).real

        # Random noise field (fixed per language instance)
        self.A_noise = rng.randn(vocab_size, vocab_size, K, K) * 0.5

        # Precompute label boundaries from flat transport norms
        # (ensures balanced classes even at ε=0)
        self._calibrate_label_boundaries(rng)

    def _calibrate_label_boundaries(self, rng, n_calibration=5000):
        """Compute label boundaries that give balanced classes at current ε."""
        norms = []
        for _ in range(n_calibration):
            seq = rng.randint(0, self.vocab_size, size=self.seq_len)
            T = self.compute_transport(seq)
            norms.append(np.linalg.norm(T, 'fro'))
        norms = np.sort(norms)
        # Quantile boundaries for balanced classes
        self.label_boundaries = [
            norms[int(len(norms) * (i + 1) / self.n_classes)]
            for i in range(self.n_classes - 1)
        ]

    def get_connection(self, v: int, w: int) -> np.ndarray:
        """Get the connection A_{v→w} = A_flat + ε · A_noise."""
        return self.A_flat[v, w] + self.epsilon * self.A_noise[v, w]

    def compute_transport(self, sequence: np.ndarray) -> np.ndarray:
        """Parallel transport along a token sequence through the GL(K) gauge field.

        Computes T = Π_{i=0}^{n-2} exp(A_{t_i → t_{i+1}}).

        Args:
            sequence: Array of token indices, shape (seq_len,).

        Returns:
            Transport matrix in GL(K), shape (K, K).
        """
        T = np.eye(self.K)
        for i in range(len(sequence) - 1):
            A = self.get_connection(int(sequence[i]), int(sequence[i + 1]))
            T = T @ expm(A)
        return T

    def compute_label(self, transport: np.ndarray) -> int:
        """Map a GL(K) transport matrix to a discrete label via Frobenius norm quantization."""
        norm = np.linalg.norm(transport, 'fro')
        label = 0
        for boundary in self.label_boundaries:
            if norm > boundary:
                label += 1
        return min(label, self.n_classes - 1)

    def generate_sample(self, rng: np.random.RandomState) -> Tuple[np.ndarray, int]:
        """Generate a single (sequence, label) pair."""
        seq = rng.randint(0, self.vocab_size, size=self.seq_len)
        T = self.compute_transport(seq)
        label = self.compute_label(T)
        return seq, label

    def measure_path_dependence(self, n_samples=1000, seed=None):
        """Measure how path-dependent the GL(K) gauge language is.

        For each sample, generates the original sequence and a permutation of
        the intermediate tokens (keeping endpoints fixed). Computes the fraction
        of samples where the label changes under permutation.

        Args:
            n_samples: Number of random sequences to test.
            seed: Random seed. Defaults to self.seed + 999.

        Returns:
            Fraction of samples whose label changed under path permutation.
            Returns 0 for flat (epsilon=0) language, increasing toward 1 for non-flat.
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed + 999)
        changes = 0
        for _ in range(n_samples):
            seq = rng.randint(0, self.vocab_size, size=self.seq_len)
            label_original = self.compute_label(self.compute_transport(seq))

            # Permute intermediate tokens (keep first and last)
            perm_seq = seq.copy()
            perm_seq[1:-1] = rng.permutation(perm_seq[1:-1])
            label_permuted = self.compute_label(self.compute_transport(perm_seq))

            if label_original != label_permuted:
                changes += 1
        return changes / n_samples


class SyntheticGaugeDataset(Dataset):
    """PyTorch Dataset wrapper for SyntheticGaugeLanguage.

    Pre-generates all samples for reproducibility. Each sample is formatted
    for autoregressive next-token prediction:
        input_ids:  [t_1, t_2, ..., t_n, SEP]
        target_ids: [t_2, ..., t_n, SEP, LABEL_TOKEN]

    where LABEL_TOKEN encodes the GL(K) holonomy classification result.
    """

    def __init__(
        self,
        language: SyntheticGaugeLanguage,
        n_samples: int = 10000,
        seed: int = 42,
        sep_token: int = None,
    ):
        self.language = language
        self.n_samples = n_samples
        self.seed = seed
        # SEP token is vocab_size (one past the last "real" token)
        self.sep_token = sep_token if sep_token is not None else language.vocab_size
        # Total vocab: vocab_size (real tokens) + 1 (SEP) + n_classes (label tokens)
        self.total_vocab_size = language.vocab_size + 1 + language.n_classes

        # Pre-generate all samples for reproducibility
        rng = np.random.RandomState(seed)
        self.samples = [language.generate_sample(rng) for _ in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        label_token = self.language.vocab_size + 1 + label  # Offset past SEP

        # Autoregressive format: [t_1, ..., t_n, SEP, LABEL_TOKEN]
        input_ids = np.concatenate([seq, [self.sep_token, label_token]])
        input_tensor = torch.tensor(input_ids, dtype=torch.long)

        return {
            'input_ids': input_tensor[:-1],   # [t_1, ..., t_n, SEP]
            'target_ids': input_tensor[1:],    # [t_2, ..., t_n, SEP, LABEL]
            'label': label,
            'sequence': torch.tensor(seq, dtype=torch.long),
        }

    @property
    def vocab_size(self):
        return self.total_vocab_size


def create_synthetic_dataloaders(
    epsilon: float = 0.0,
    vocab_size: int = 64,
    K: int = 3,
    n_classes: int = 4,
    seq_len: int = 16,
    n_train: int = 50000,
    n_val: int = 5000,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val dataloaders for the synthetic gauge language.

    Args:
        epsilon: Holonomy strength. 0 = flat (path-independent), >0 = non-flat.
        vocab_size: Number of tokens in the synthetic vocabulary.
        K: Dimension of the gauge group GL(K).
        n_classes: Number of discrete label classes for classification.
        seq_len: Length of generated token sequences.
        n_train: Number of training samples to generate.
        n_val: Number of validation samples to generate.
        batch_size: Batch size for DataLoaders.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, total_vocab_size).
    """
    language = SyntheticGaugeLanguage(
        vocab_size=vocab_size, K=K, epsilon=epsilon,
        n_classes=n_classes, seq_len=seq_len, seed=seed,
    )

    train_dataset = SyntheticGaugeDataset(language, n_samples=n_train, seed=seed)
    val_dataset = SyntheticGaugeDataset(language, n_samples=n_val, seed=seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.vocab_size
