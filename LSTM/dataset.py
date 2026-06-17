from datasets import load_dataset, interleave_datasets, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader

# Per-GPU batch size — main.py multiplies this by NUM_GPUS
PER_GPU_BATCH_SIZE = 256

# configs = ['ka_ge', 'fi_fi', 'ar_eg', 'sw_ke', 'ta_in', 'th_th', 'cmn_hans_cn']
configs = ['my_mm', 'ceb_ph', 'fil_ph', 'id_id', 'jv_id', 'km_kh', 'lo_la', 'ms_my', 'mi_nz', 'th_th', 'vi_vn']

# Raw FLEURS lang_id integers for the target languages, in the same order as configs
# LANG_IDS = [42, 24, 2, 86, 87, 90, 13]
LANG_IDS = [11, 25, 36, 41, 46, 53, 57, 62, 64, 90, 96]

assert len(LANG_IDS) == len(configs)

# Map lang_id integer -> class index (0-15)
label2id = {lang_id: idx for idx, lang_id in enumerate(LANG_IDS)}

# Map class index -> human-readable config string (useful for per-class reporting)
id2config = {idx: cfg for idx, cfg in enumerate(configs)}

NUM_CLASSES = len(configs)


def load_fleurs_dataset(
    train_collate_fn=None,
    val_collate_fn=None,
    test_collate_fn=None,
    batch_size=PER_GPU_BATCH_SIZE,
):
    """
    Downloads (or loads from cache) all 16 language configs, interleaves
    their splits, shuffles, and returns (train_loader, val_loader, test_loader).
    """
    datasetdicts = []
    for cfg in configs:
        print(f"  Loading {cfg} ...")
        datasetdicts.append(load_dataset("google/fleurs", cfg, cache_dir="./datasets", trust_remote_code=True))

    combined = DatasetDict({
        split: interleave_datasets(
            [ds[split] for ds in datasetdicts],
            stopping_strategy="first_exhausted",  # cycle shorter splits; use all data
        )
        for split in datasetdicts[0].keys()
    })
    
    # Recombine all datasets
    
    # all_ds = concatenate_datasets([combined["train"], combined["validation"], combined["test"]])
    # all_ds = all_ds.shuffle(seed=0)
    
    # Resplit dataset
    
    # tmp = all_ds.train_test_split(test_size=0.10, seed=0)  # 10% test
    # tmp2 = tmp["train"].train_test_split(test_size=0.10, seed=0)  # 10% of remaining -> val

    # resplit = DatasetDict({
    #     "train": tmp2["train"],          # 81%
    #     "validation": tmp2["test"],      # 9%
    #     "test": tmp["test"],             # 10%
    # })

    # Shuffle each split individually — DatasetDict.shuffle() is unreliable
    # combined = DatasetDict({split: ds.shuffle(seed=42) for split, ds in combined.items()})

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

    return train_loader, val_loader, test_loader
