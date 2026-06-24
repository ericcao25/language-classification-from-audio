# Audio Speech Recognition - CS172B Final Project

This project explores language classification across various linguistic regions using four different model architectures:
1. **Feed-Forward Neural Networks (FFN)**
2. **Convolutional Neural Networks (CNN)**
3. **Long Short-Term Memory Networks (LSTM)**
4. **Transformers: wav2vec**

We compare the efficacy of these four model types on an audio speech recognition dataset to classify languages based on audio samples.

## Report
For an in-depth discussion of the models and results, read the [report](report.pdf).

## Project Structure

- **`FFN/`**: Python scripts for the Feed-Forward Neural Network models.
- **`CNN/`**: Python scripts for training and evaluating CNN models on different regional datasets.
- **`LSTM/`**: Python scripts for defining, training, and testing LSTM models.
- **`transformer/`**: Python scripts for the Transformer model architecture.
- **`build_configs.py`**: Contains region definitions (language configs, IDs, names) used by all models.
- **`dataset_analysis.ipynb`**: Used for initial dataset exploration and analysis.

## Setting up the environment

Requires Python 3.12. Install all dependencies using:
```
pip install -r requirements.txt
```

## Notes

Before running, set a Hugging Face token to increase dataset streaming speed:
```
export HF_TOKEN=your_token_here
```

Running the CNN, LSTM, and Transformer on a GPU is strongly recommended, as training on CPU is very slow.

## Running the FFN

Valid regions: `eastern_europe`, `western_europe`, `south_asia`, `central_asia_middle_east_north_africa`, `sub_saharan_africa`, `south_east_asia`, `cjk`

```
cd FFN
python train.py --region eastern_europe
```

If shards for the specified region don't already exist, preprocessing will run automatically before training. Optional flags:
```
python train.py --region eastern_europe --batch_size 256 --save_root fleurs_preprocessed
```

## Running the CNN

Valid regions: `eastern_europe`, `western_europe`, `south_asia`, `central_asia_middle_east_north_africa`, `sub_saharan_africa`, `south_east_asia`, `cjk`

```
cd CNN
python train.py --region eastern_europe
```

Optional flags:
```
python train.py --region eastern_europe --num_epochs 5 --batch_size 64
```

## Running the LSTM

Valid regions: `eastern_europe`, `western_europe`, `south_asia`, `central_asia_middle_east_north_africa`, `sub_saharan_africa`, `south_east_asia`, `cjk`

```
cd LSTM
python main.py --region eastern_europe
```

Optional flags:
```
python main.py --region eastern_europe --num_epochs 100 --batch_size 64
```

## Running the Transformer

Valid regions: `eastern_europe`, `western_europe`, `south_asia`, `central_asia_middle_east_north_africa`, `sub_saharan_africa`, `south_east_asia`, `cjk`

**Training:**
```
cd transformer
python main.py --region eastern_europe
```

Optional flags:
```
python main.py --region eastern_europe --num_epochs 1 --pooling_mode mean --train_batch_size 2 --eval_batch_size 4
```

**Evaluation** (requires a trained checkpoint at `{region}/best.pt`):
```
python eval.py --region eastern_europe
```

Optional flags:
```
python eval.py --region eastern_europe --pooling_mode mean
```
