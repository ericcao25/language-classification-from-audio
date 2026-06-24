import sys
import torch
import torchaudio
from datasets import load_dataset, interleave_datasets, DatasetDict, concatenate_datasets
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_configs import FLEURS_GROUP_INFO


TARGET_SR = 16000
N_MELS = 80

def make_transforms():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=400,
        hop_length=160,
        n_mels=N_MELS,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
    return mel, amp_to_db, freq_mask, time_mask

def _waveform_to_logmel(waveform, mel, amp_to_db):
    m = mel(waveform)       # [1, F, T]
    m = amp_to_db(m)        # log scale
    return m                 # [1, F, T]

def collate_fn(batch, label2id, train=False):
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
    mel, amp_to_db, freq_mask, time_mask = make_transforms()
    for ex in batch:
        assert int(ex["audio"]["sampling_rate"]) == TARGET_SR, \
            f"Unexpected sample rate: {ex['audio']['sampling_rate']}"
        assert ex["lang_id"] in label2id, \
            f"Unknown lang_id {ex['lang_id']} — not in label2id mapping"

        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)  # [1, N]

        m = _waveform_to_logmel(wav, mel, amp_to_db)  # [1, F, T]

        # Per-utterance normalisation BEFORE augmentation so that
        # masked (zeroed) regions don't corrupt the mean/std estimate
        m = (m - m.mean()) / (m.std() + 1e-5)

        if train:
            m = freq_mask(m)
            m = time_mask(m)

        feats = m.squeeze(0).transpose(0, 1).contiguous()  # [T, F]

        feats_list.append(feats)
        lengths.append(feats.shape[0])
        target_labels.append(label2id[ex["lang_id"]])

    padded        = pad_sequence(feats_list, batch_first=True)          # [B, Tmax, F]
    lengths       = torch.tensor(lengths,       dtype=torch.long)
    target_labels = torch.tensor(target_labels, dtype=torch.long)

    return padded, lengths, target_labels

def load_fleurs_dataset(region, batch_size=64):
    """
    Downloads (or loads from cache) all 16 language configs, interleaves
    their splits, shuffles, and returns (train_loader, val_loader, test_loader).
    """
    configs = FLEURS_GROUP_INFO[region]["configs"]
    lang_ids = FLEURS_GROUP_INFO[region]["lang_ids"]
    label2id = {lang_id: idx for idx, lang_id in enumerate(lang_ids)}
    id2config = {idx: cfg for idx, cfg in enumerate(configs)}

    datasetdicts = []
    for cfg in configs:
        print(f"  Loading {cfg} ...")
        datasetdicts.append(load_dataset("google/fleurs", cfg, trust_remote_code=True))

    combined = DatasetDict({
        split: interleave_datasets(
            [ds[split] for ds in datasetdicts],
            stopping_strategy="first_exhausted",  # cycle shorter splits; use all data
        )
        for split in datasetdicts[0].keys()
    })
    train_collate_fn = partial(collate_fn, label2id=label2id, train=True)
    val_collate_fn = partial(collate_fn, label2id=label2id, train=False)
    test_collate_fn = partial(collate_fn, label2id=label2id, train=False)

    train_loader = DataLoader(
        # resplit["train"],
        combined["train"],
        batch_size=batch_size,
        collate_fn=train_collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        # resplit["validation"],
        combined["validation"],
        batch_size=batch_size,
        collate_fn=val_collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        # resplit["test"],
        combined["test"],
        batch_size=batch_size,
        collate_fn=test_collate_fn,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader, id2config
