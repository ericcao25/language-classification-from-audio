import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any

from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoConfig, AutoFeatureExtractor
from transformers.utils import ModelOutput
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model


# ----------------------------
# 1) Choose languages (FLEURS configs)
# ----------------------------
# LANG_CONFIGS = ['hy_am', 'be_by', 'bg_bg', 'cs_cz', 'et_ee', 'ka_ge', 'lv_lv', 'lt_lt', 'mk_mk', 'pl_pl', 'ro_ro', 'ru_ru', 'sr_rs', 'sk_sk', 'sl_si', 'uk_ua']
LANG_CONFIGS = ['my_mm', 'ceb_ph', 'fil_ph', 'id_id', 'jv_id', 'km_kh', 'lo_la', 'ms_my', 'mi_nz', 'th_th', 'vi_vn']
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
POOLING_MODE = "mean"  # "mean" / "sum" / "max"

MAX_SEC = 10.0
MAX_SAMPLES = int(16000 * MAX_SEC)

# ----------------------------
# 2) Build FLEURS train/val across configs
# ----------------------------

def crop_audio(ex):
    a = ex["audio"]["array"]
    if len(a) > MAX_SAMPLES:
        a = a[:MAX_SAMPLES]
    ex["audio"]["array"] = a
    return ex

def load_fleurs_langid_dataset(lang_configs: List[str], sampling_rate: int = 16000):
    train_parts = []
    val_parts = []
    test_parts = []

    for cfg in lang_configs:
        ds = load_dataset("google/fleurs", cfg)
        # Optional but recommended: standardize sampling rate in the dataset
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

        train_parts.append(ds["train"].map(lambda x, cfg=cfg: {"label_str": cfg}))
        val_parts.append(ds["validation"].map(lambda x, cfg=cfg: {"label_str": cfg}))
        test_parts.append(ds["test"].map(lambda x, cfg=cfg: {"label_str": cfg}))

    train_ds = concatenate_datasets(train_parts).shuffle(seed=0)
    val_ds = concatenate_datasets(val_parts)
    test_ds = concatenate_datasets(test_parts)

    label2id = {l: i for i, l in enumerate(lang_configs)}
    id2label = {i: l for l, i in label2id.items()}

    def to_id(batch):
        batch["label"] = label2id[batch["label_str"]]
        return batch
    
    # Crop audio to prevent exceeding memory
    # train_ds = train_ds.map(crop_audio)
    # val_ds   = val_ds.map(crop_audio)
    # test_ds  = test_ds.map(crop_audio)

    train_ds = train_ds.map(to_id)
    val_ds = val_ds.map(to_id)
    test_ds = test_ds.map(to_id)

    return train_ds, val_ds, test_ds, label2id, id2label


# ----------------------------
# 3) Preprocess (feature extractor -> input_values)
# ----------------------------
def make_preprocess_fn(feature_extractor):
    def preprocess(batch):
        audio = batch["audio"]
        x = audio["array"]

        # Make mono: (1, T) -> (T,)
        if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 1:
            x = x[0]

        out = feature_extractor(
            x,
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        out["labels"] = batch["label"]
        return out

    return preprocess


# ----------------------------
# 4) Model: wav2vec2 + pooling + classifier
# ----------------------------
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        # Compatibility shim for some transformers versions
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        return {k: True for k in keys}

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, attention_mask=None, mode="mean"):
        # hidden_states: (B, T, H)
        if mode == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (B, T, 1)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        elif mode == "sum":
            if attention_mask is None:
                return hidden_states.sum(dim=1)
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            return (hidden_states * mask).sum(dim=1)
        elif mode == "max":
            if attention_mask is None:
                return hidden_states.max(dim=1)[0]
            # set padded positions to -inf before max
            mask = attention_mask.unsqueeze(-1).bool()
            neg_inf = torch.finfo(hidden_states.dtype).min
            masked = hidden_states.masked_fill(~mask, neg_inf)
            return masked.max(dim=1)[0]
        raise ValueError("pooling_mode must be one of ['mean','sum','max']")

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        pooled = self.merged_strategy(hidden_states, None, self.pooling_mode)
        logits = self.classifier(pooled)
        return logits


# ----------------------------
# 5) Collator: pad inputs, stack labels
# ----------------------------
@dataclass
class DataCollatorSpeechClassificationWithPadding:
    feature_extractor: Any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            iv = f["input_values"]

            # iv can be:
            # - list[float]                 (good)
            # - list[list[float]] with shape [1][T] (bad)
            # - numpy arrays sometimes
            if isinstance(iv, np.ndarray):
                if iv.ndim == 2 and iv.shape[0] == 1:
                    iv = iv[0]
                iv = iv.tolist()
            else:
                # python lists
                if len(iv) > 0 and isinstance(iv[0], (list, tuple, np.ndarray)):
                    # assume shape [1][T]
                    iv = iv[0]

            item = {"input_values": iv}

            am = f.get("attention_mask", None)
            if am is not None:
                # also flatten mask if it is [1][T]
                if isinstance(am, np.ndarray):
                    if am.ndim == 2 and am.shape[0] == 1:
                        am = am[0]
                    am = am.tolist()
                else:
                    if len(am) > 0 and isinstance(am[0], (list, tuple, np.ndarray)):
                        am = am[0]
                item["attention_mask"] = am

            input_features.append(item)

        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        batch = self.feature_extractor.pad(
            input_features,
            padding=True,  # force padding on
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


# ----------------------------
# 6) Training + evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=True):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_values=input_values, attention_mask=attention_mask)
            loss = ce(logits, labels)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        loss_sum += loss.item() * labels.size(0)

    return {
        "eval_loss": loss_sum / max(total, 1),
        "eval_acc": correct / max(total, 1),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, best_metric=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "best_metric": best_metric,
        },
        path,
    )


def main():
    # ----------------------------
    # Hyperparams (match your Trainer-ish setup)
    # ----------------------------
    output_dir = "xlsr300m-fleurs-langid"
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 4
    grad_accum_steps = 2
    num_epochs = 1
    lr = 2e-5
    eval_every_steps = 500
    save_every_steps = 500
    use_amp = True

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    target_sr = 16000

    train_ds, val_ds, _, label2id, id2label = load_fleurs_langid_dataset(LANG_CONFIGS, sampling_rate=target_sr)

    preprocess = make_preprocess_fn(feature_extractor)
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LANG_CONFIGS),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_langid",
    )
    setattr(config, "pooling_mode", POOLING_MODE)

    model = Wav2Vec2ForSpeechClassification.from_pretrained(MODEL_NAME, config=config)
    model.freeze_feature_extractor()
    model.to(device)

    collator = DataCollatorSpeechClassificationWithPadding(feature_extractor=feature_extractor)

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # simple linear warmup optional: set warmup_ratio=0.1-ish if you want
    total_updates = math.ceil(len(train_loader) / grad_accum_steps) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss()

    global_step = 0
    best_acc = -1.0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_values=input_values, attention_mask=attention_mask)
                loss = ce(logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # scheduler steps per optimizer update
                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                if global_step % 50 == 0:
                    # unscaled loss displayed (multiply back)
                    print(f"[epoch {epoch+1}/{num_epochs} | step {global_step}] loss={loss.item()*grad_accum_steps:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

                if eval_every_steps and (global_step % eval_every_steps == 0):
                    metrics = evaluate(model, val_loader, device, use_amp=use_amp)
                    print(f"[eval @ step {global_step}] loss={metrics['eval_loss']:.4f} acc={metrics['eval_acc']:.4f}")

                    # best checkpoint
                    if metrics["eval_acc"] > best_acc:
                        best_acc = metrics["eval_acc"]
                        save_checkpoint(
                            os.path.join(output_dir, "best.pt"),
                            model, optimizer, scheduler, scaler,
                            step=global_step,
                            best_metric=best_acc,
                        )
                        print(f"  -> new best acc: {best_acc:.4f} (saved best.pt)")

                if save_every_steps and (global_step % save_every_steps == 0):
                    save_checkpoint(
                        os.path.join(output_dir, f"ckpt_step{global_step}.pt"),
                        model, optimizer, scheduler, scaler,
                        step=global_step,
                        best_metric=best_acc,
                    )
                    print(f"  -> saved checkpoint ckpt_step{global_step}.pt")

        # epoch-end eval
        metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        print(f"[epoch-end eval {epoch+1}] loss={metrics['eval_loss']:.4f} acc={metrics['eval_acc']:.4f}")

    # final eval
    metrics = evaluate(model, val_loader, device, use_amp=use_amp)
    print(f"[final eval] loss={metrics['eval_loss']:.4f} acc={metrics['eval_acc']:.4f}")


if __name__ == "__main__":
    main()