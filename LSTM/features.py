"""
features.py — audio feature extraction and collation.

Kept separate from dataset.py (data loading) and main.py (training orchestration)
so each file has a single clear responsibility.
"""

from functools import partial

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from dataset import label2id

TARGET_SR = 16000  # FLEURS is always 16 kHz
N_MELS    = 80

# ---------------------------------------------------------------------------
# Transforms (module-level so they are instantiated once and reused)
# ---------------------------------------------------------------------------
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=400,
    hop_length=160,
    n_mels=N_MELS,
)
_amp_to_db = torchaudio.transforms.AmplitudeToDB()

# SpecAugment parameters (applied during training only)
_freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
_time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)


def _waveform_to_logmel(waveform: torch.Tensor) -> torch.Tensor:
    """
    waveform : [1, N] float32 mono, 16 kHz
    returns  : [1, F, T] log-mel spectrogram
    """
    m = _mel(waveform)       # [1, F, T]
    m = _amp_to_db(m)        # log scale
    return m                 # [1, F, T]


def collate_fn(batch, train: bool = False):
    """
    batch : list of FLEURS examples (dicts with 'audio', 'lang_id', ...)
    train : if True, apply SpecAugment

    Returns (padded, lengths, labels):
        padded  : [B, Tmax, N_MELS]  float32
        lengths : [B]                int64
        labels  : [B]                int64  (class indices 0-15)
    """
    feats_list    = []
    lengths       = []
    target_labels = []

    for ex in batch:
        assert int(ex["audio"]["sampling_rate"]) == TARGET_SR, \
            f"Unexpected sample rate: {ex['audio']['sampling_rate']}"
        assert ex["lang_id"] in label2id, \
            f"Unknown lang_id {ex['lang_id']} — not in label2id mapping"

        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)  # [1, N]

        m = _waveform_to_logmel(wav)  # [1, F, T]

        # Per-utterance normalisation BEFORE augmentation so that
        # masked (zeroed) regions don't corrupt the mean/std estimate
        m = (m - m.mean()) / (m.std() + 1e-5)

        if train:
            m = _freq_mask(m)
            m = _time_mask(m)

        feats = m.squeeze(0).transpose(0, 1).contiguous()  # [T, F]

        feats_list.append(feats)
        lengths.append(feats.shape[0])
        target_labels.append(label2id[ex["lang_id"]])

    padded        = pad_sequence(feats_list, batch_first=True)          # [B, Tmax, F]
    lengths       = torch.tensor(lengths,       dtype=torch.long)
    target_labels = torch.tensor(target_labels, dtype=torch.long)

    return padded, lengths, target_labels


# Ready-made partials for passing to the DataLoader
train_collate_fn = partial(collate_fn, train=True)
val_collate_fn   = partial(collate_fn, train=False)
test_collate_fn  = partial(collate_fn, train=False)
