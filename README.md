# Audio Speech Recognition - CS172B Final Project

This project explores language classification across various linguistic regions using four different model architectures:
1. **Convolutional Neural Networks (CNN)**
2. **Feed-Forward Neural Networks (FFN)**
3. **Long Short-Term Memory Networks (LSTM)**
4. **Transformers**

We compare the efficacy of these four model types on an audio speech recognition dataset to classify languages based on audio samples.

## Project Structure

- **`CNN/`**: Contains notebooks for training and evaluating CNN models on different regional datasets.
- **`FFN/`**: Contains notebooks for the Feed-Forward Neural Network models.
- **`LSTM/`**: Python scripts for defining, training, and testing LSTM models.
- **`transformer/`**: Python scripts for the Transformer model architecture.
- **Preprocessing & Analysis Notebooks** (Root Directory):
  - `Dataset Analysis.ipynb`: Used for initial dataset exploration and analysis.
  - `PreprocessingEastern_European.ipynb`: Data processing for the Eastern European language group.
  - `PreprocessingSouthAsian.ipynb`: Data processing for the South Asian language group.

## Setting up the environment

Need Python 3.14.2

Install all library requirements using:
```
pip install -r requirements.txt
```

## Notes
Before running, you should set a hugging face token using:
```
export HF_TOKEN=your_token_here
```
This increases the streaming speed when accessing the dataset

## Example dataset output

Dataset is in the format:
```
{'id': 845, 'num_samples': 270720, 'path': None, 'audio': {'path': 'train/14699537390019605585.wav', 
'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00014174,
       -0.00013965, -0.00014967], shape=(270720,)), 'sampling_rate': 16000}, 
       'transcription': 'जरी ते आधुनिक घटनांपासून खूप दूर असले तरी बरेच लोक कदाचित त्यास स्प्रे पेंट वापरून सार्वजनिक आणि खाजगी मालमत्तेचा विध्वंस करणाऱ्या तरुणांशी जोडतात', 
       'raw_transcription': 'जरी ते आधुनिक घटनांपासून खूप दूर असले, तरी बरेच लोक कदाचित त्यास स्प्रे पेंट वापरून सार्वजनिक आणि खाजगी मालमत्तेचा विध्वंस करणाऱ्या तरुणांशी जोडतात.', 
       'gender': 0, 'lang_id': 0, 'language': 'Afrikaans', 'lang_group_id': 3}
```