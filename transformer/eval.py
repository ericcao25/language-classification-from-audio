import argparse
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor

import transformer.train as tm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_configs import FLEURS_GROUP_INFO

def load_model(checkpoint_path: str, configs: list[str], pooling_mode: str, device: torch.device):
    # build config consistent with training
    config = AutoConfig.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        num_labels=len(configs),
        label2id={l: i for i, l in enumerate(configs)},
        id2label={i: l for i, l in enumerate(configs)},
        finetuning_task="wav2vec2_langid",
    )
    setattr(config, "pooling_mode", pooling_mode)

    # instantiate model from pretrained backbone and then load checkpoint weights
    model = tm.Wav2Vec2ForSpeechClassification.from_pretrained("facebook/wav2vec2-xls-r-300m", config=config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    model.to(device)
    return model


def main(region, pooling_mode="mean"):
    configs = FLEURS_GROUP_INFO[region]["configs"]
    checkpoint = f"{region}/best.pt"
    batch_size = 8
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # feature extractor & dataset
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    target_sr = 16000

    _, _, test_ds, _, _ = tm.load_fleurs_langid_dataset(configs, sampling_rate=target_sr)
    preprocess = tm.make_preprocess_fn(feature_extractor)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)

    collator = tm.DataCollatorSpeechClassificationWithPadding(feature_extractor=feature_extractor)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )

    model = load_model(checkpoint, configs, pooling_mode, device)
    use_amp = True if device == "cuda" else False
    
    model.eval()

    metrics = tm.evaluate(model, test_loader, device, use_amp=use_amp)
    print(f"[test] loss={metrics['eval_loss']:.4f} acc={metrics['eval_acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLEURS regional language-ID pipeline")
    parser.add_argument("--region", type=str, choices=["eastern_europe", "western_europe", "central_asia_middle_east_north_africa", "sub_saharan_africa", "south_asia", "south_east_asia", "cjk"])
    parser.add_argument("--pooling_mode", type=str, choices=["mean", "sum", "max"], default="mean")
    args = parser.parse_args()
    main(args.region, pooling_mode=args.pooling_mode)