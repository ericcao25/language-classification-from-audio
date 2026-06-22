import os
import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets, DatasetDict
from build_configs import load_configs


TARGET_SR = 16000
N_MELS = 80

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

def waveform_to_logmel(waveform_1xN, mel, amp_to_db):
    m = mel(waveform_1xN)                         # [1, 80, T]
    lm = amp_to_db(m)                             # [1, 80, T]
    return lm.squeeze(0).transpose(0, 1).contiguous()  # [T, 80]

def preprocess_and_save_shards(split_ds, shard_dir, prefix, global_to_local, 
                               mel, amp_to_db, shard_size=500):
    os.makedirs(shard_dir, exist_ok=True)

    shard = []
    shard_idx = 0
    total = 0

    for ex in tqdm(split_ds):
        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        gid = int(ex["lang_id"])
        lid = global_to_local[gid]  # 0..15

        x = waveform_to_logmel(wav.unsqueeze(0), mel, amp_to_db)  # [T,80]

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

def preprocess(region, save_root="fleurs_preprocessed"):
    region_info = load_configs(region)
    global_ids = region_info["lang_ids"]
    configs = region_info["configs"]
    ds = load_group_dataset_non_streaming(configs)
    global_to_local = {gid: i for i, gid in enumerate(global_ids)}

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=400,
        hop_length=160,
        n_mels=N_MELS,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    save_root += "/" + region
    os.makedirs(save_root, exist_ok=True)
    train_shards = os.path.join(save_root, "train_shards")
    val_shards   = os.path.join(save_root, "validation_shards")
    test_shards  = os.path.join(save_root, "test_shards")
    preprocess_and_save_shards(ds["train"], train_shards, "train", global_to_local, mel, amp_to_db, shard_size=500)
    preprocess_and_save_shards(ds["validation"], val_shards, "validation", global_to_local, mel, amp_to_db, shard_size=500)
    preprocess_and_save_shards(ds["test"], test_shards, "test", global_to_local, mel, amp_to_db, shard_size=500)
    

if __name__ == "__main__":
    preprocess("eastern_europe")