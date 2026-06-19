import datasets
import os
import math
import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets, DatasetDict


EE_CONFIGS = [
    "hy_am","be_by","bg_bg","cs_cz","et_ee","ka_ge","lv_lv","lt_lt",
    "mk_mk","pl_pl","ro_ro","ru_ru","sr_rs","sk_sk","sl_si","uk_ua"
]

EE_GLOBAL_IDS = [35, 6, 7, 14, 21, 42, 56, 54, 58, 74, 77, 78, 84, 80, 81, 92]
global_to_local = {gid: i for i, gid in enumerate(EE_GLOBAL_IDS)}

def load_group_dataset_non_streaming(configs):
    ds_list = []
    for cfg in configs:
        print(f"Loading config: {cfg}")
        ds_list.append(load_dataset("google/fleurs", cfg))

    combined = DatasetDict({
        split: interleave_datasets([ds[split] for ds in ds_list])
        for split in ds_list[0].keys()
    })

    combined["train"] = combined["train"].shuffle(seed=42)
    combined["validation"] = combined["validation"].shuffle(seed=42)

    return combined

ee_ds = load_group_dataset_non_streaming(EE_CONFIGS)

TARGET_SR = 16000
N_MELS = 80

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=400,
    hop_length=160,
    n_mels=N_MELS,
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def waveform_to_logmel(waveform_1xN: torch.Tensor) -> torch.Tensor:
    m = mel(waveform_1xN)                         # [1, 80, T]
    lm = amp_to_db(m)                             # [1, 80, T]
    return lm.squeeze(0).transpose(0, 1).contiguous()  # [T, 80]


SAVE_ROOT = "/fleurs_preprocessed/eastern_european"
os.makedirs(SAVE_ROOT, exist_ok=True)

def preprocess_and_save_shards(split_ds, shard_dir, prefix, shard_size=500):
    os.makedirs(shard_dir, exist_ok=True)

    shard = []
    shard_idx = 0
    total = 0

    for ex in tqdm(split_ds):
        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        gid = int(ex["lang_id"])
        lid = global_to_local[gid]  # 0..15

        x = waveform_to_logmel(wav.unsqueeze(0))  # [T,80]

        shard.append({
            "x": x,
            "y": lid,
            "global_lang_id": gid,
            "language": ex.get("language", "N/A"),
        })
        total += 1

        if len(shard) >= shard_size:
            out_path = os.path.join(shard_dir, f"{prefix}_shard_{shard_idx:03d}.pt")
            torch.save(shard, out_path)
            shard = []
            shard_idx += 1

    # save remainder
    if shard:
        out_path = os.path.join(shard_dir, f"{prefix}_shard_{shard_idx:03d}.pt")
        torch.save(shard, out_path)

    print(f"Saved {total} examples into shards at: {shard_dir}")

train_shards = os.path.join(SAVE_ROOT, "train_shards")
val_shards   = os.path.join(SAVE_ROOT, "validation_shards")
test_shards  = os.path.join(SAVE_ROOT, "test_shards")

preprocess_and_save_shards(ee_ds["train"], train_shards, "train", shard_size=500)
preprocess_and_save_shards(ee_ds["validation"], val_shards, "validation", shard_size=500)
preprocess_and_save_shards(ee_ds["test"], test_shards, "test", shard_size=500)

def main():


if __name__ == "__main__":
    main()