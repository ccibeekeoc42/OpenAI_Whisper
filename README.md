# Advancing Multilingual Speech Recognition: Fine-Tuning OpenAI Whisper for Enhanced Low-Resource Performance

## Introduction
Based on the [blog post](https://medium.com/@ccibeekeoc42/advancing-multilingual-speech-recognition-fine-tuning-whisper-for-enhanced-low-resource-34529b525f90), this repository hosts the fine-tuning framework for the Whisper model, a state-of-the-art, multilingual speech recognition system developed by OpenAI. Our focus is on enhancing performance for low-resource languages, specifically Yoruba, using strategic data inclusion, parameter-efficient techniques, and efficient hardware configurations.

## The Whisper Model: An Overview
Whisper, developed by OpenAI, is a robust speech recognition system trained on 680,000 hours of audio data. Key features include:
- **Model Architecture**: Transformer encoder-decoder structure optimized for large-scale weak supervision.
- **Large-Scale Weak Supervision**: Leveraging massive datasets to generalize across diverse audio sources.
- **Multilingual and Multitask Training**: Handling multiple languages and tasks within a single framework.
- **Robustness to Distribution Shifts**: Maintaining high performance across different datasets and conditions.
- **Scaling Laws and Performance Trends**: Demonstrating performance improvements with increasing dataset size and model parameters.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- Transformers 4.8+
- Datasets library from Hugging Face

### Setup
Clone this repository and install the required packages:
```bash
git clone https://github.com/ccibeekeoc42/openAI_Whisper.git
cd openAI_Whisper
```

## Usage
To understand and replicate this work on your dataset, follow the steps outlined below:

### Data Preparation & Model Training (Fine-Tuning)
Refer to the file in this repo called "Whisper_Finetune.ipynb" to run the fine-tuning process adapted to your needs.

### Inference/Evaluation
Refer to the file in this repo called "Default_WER.ipynb" to evaluate the fine-tuned model and understand its performance and improvements.

## Technical Workflow: A Deep Dive

### Dataset Acquisition & Preparation
Fine-tuning Whisper for low-resource ASR required a curated dataset, including:
- **Common Voice Dataset**: 31,175 hours of audio with corresponding text files, focusing on the Yoruba subset.
- **Google Fleurs Dataset**: 2,009 n-way parallel sentences in 102 languages, focusing on Yoruba.
- **Miscellaneous Datasets**: Smaller supervised datasets like OpenSLR-Yoruba and Yoruba-Audio-Data.
- **TED-LIUM Dataset**: English-language TED talks with transcriptions.
- **LibriSpeech Dataset**: 1,000 hours of read English speech.

### Dataset Analysis & Exploration
We grouped our fine-tuning data into two configurations:
1. **Run 1 (Yoruba Only Data)**: Yoruba data from Common Voice, Fleurs, and miscellaneous datasets.
2. **Run 2 (Yoruba + English Data)**: A mix of Yoruba and English data from Common Voice, Fleurs, LibriSpeech, and TED-LIUM.

### Whisper Default Model Analysis
We evaluated the default Whisper model using Word Error Rate (WER) as the key metric. The Whisper-small checkpoint (244M parameters) served as our base model, achieving an average of 11% and 50% WERs on English and multilingual speech recognition tasks, respectively.

### Model Loading & Fine-Tuning
We loaded the default openai/whisper-small model and optimized it for both memory usage and computational efficiency during fine-tuning. Key configurations included:
- **Reduced Precision for Efficiency**: Using FP16 precision.
- **Standard Attention Implementation**: Utilizing the standard attention mechanism.
- **Gradient Checkpointing**: Balancing memory savings with computational efficiency.
- **Disabled Caching**: Disabled during fine-tuning but re-enabled for predictions.

Fine-tuning was conducted using a custom training loop on a single L4 GPU on Google Colab paid tier.

## Results: A Path Towards Under-Represented ASR Systems
Our fine-tuning process yielded significant improvements:
- **Run 1 (Yoruba Only Data)**: Improved WER for Yoruba but suffered from catastrophic forgetting in English data.
- **Run 2 (Yoruba + English Data)**: Achieved balanced improvements across both languages, mitigating catastrophic forgetting.

## Contributing
Contributions to this project are welcome! You can contribute in several ways:
1. **Issues**: Submit issues for any bugs encountered or enhancements.
2. **Pull Requests**: Submit PRs for bug fixes or new features, following the pull request guidelines provided.
3. **Documentation**: Enhancements to documentation or new examples are always appreciated.

## License
This project is licensed under the terms of the Apache license.

## Citation
If you use this framework or the Whisper model in your research, please cite it as follows:
```bibtex
@misc{whisper_2024,
  title={Advancing Multilingual Speech Recognition: Fine-Tuning Whisper for Enhanced Low-Resource Performance},
  author={Christopher Ibe & Okezie Okoye},
  year={2024},
  howpublished={\url{https://github.com/ccibeekeoc42/OpenAI_Whisper}},
}
```

## About the Authors
Christopher Ibe and Okezie Okoye lead Hypa AI in advancing AI translation, promoting inclusivity, and celebrating linguistic diversity.

Hypa AI and its subsidiary, AfroVoices, remain committed to bridging the digital representation gap and amplifying African voices in the intelligence age.
