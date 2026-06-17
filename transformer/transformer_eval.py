import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor

import transformer.transformer_main as tm


def load_model(checkpoint_path: str, device: torch.device):
    # build config consistent with training
    config = AutoConfig.from_pretrained(
        tm.MODEL_NAME,
        num_labels=len(tm.LANG_CONFIGS),
        label2id={l: i for i, l in enumerate(tm.LANG_CONFIGS)},
        id2label={i: l for i, l in enumerate(tm.LANG_CONFIGS)},
        finetuning_task="wav2vec2_langid",
    )
    setattr(config, "pooling_mode", tm.POOLING_MODE)

    # instantiate model from pretrained backbone and then load checkpoint weights
    model = tm.Wav2Vec2ForSpeechClassification.from_pretrained(tm.MODEL_NAME, config=config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    model.to(device)
    return model


def main():
    checkpoint = "sa_xlsr300m-fleurs-langid/best.pt"
    batch_size = 8
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # feature extractor & dataset
    feature_extractor = AutoFeatureExtractor.from_pretrained(tm.MODEL_NAME)
    target_sr = 16000

    _, _, test_ds, _, _ = tm.load_fleurs_langid_dataset(tm.LANG_CONFIGS, sampling_rate=target_sr)
    preprocess = tm.make_preprocess_fn(feature_extractor)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)

    collator = tm.DataCollatorSpeechClassificationWithPadding(feature_extractor=feature_extractor)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    model = load_model(checkpoint, device)
    use_amp = True if device == "cuda" else False
    
    model.eval()

    metrics = tm.evaluate(model, test_loader, device, use_amp=use_amp)
    print(f"[test] loss={metrics['eval_loss']:.4f} acc={metrics['eval_acc']:.4f}")


if __name__ == "__main__":
    main()