# Major-Project-on-Generating-Descriptive-Image-Captions-Using-Deep-Learning
📸✨ Image Caption Generator
Deep Learning–Based System for Generating Descriptive Image Captions
🚀 Overview

This project demonstrates a complete deep-learning pipeline that converts images into meaningful human-like captions. By combining InceptionV3 (CNN) for vision and LSTM for language generation—enhanced with attention mechanisms—the system learns to “see” an image and “describe” it in natural language.

It is designed, trained, and fine-tuned on the MS COCO dataset, achieving strong performance with coherent, context-aware captions.

🧠 Key Features

✔️ Encoder–Decoder Architecture (InceptionV3 + LSTM)
✔️ Attention Mechanism for focus on key image regions
✔️ End-to-End Training with captions + images
✔️ Fine-Tuning of CNN for improved visual understanding
✔️ BLEU Score Evaluation (Quantitative & Qualitative)
✔️ User-Friendly Code Structure (Feature extraction, tokenization, training pipeline)

📂 Dataset — MS COCO

330,000+ images

1.5M+ object instances

80 object categories, 91 “stuff” categories

5 human-written captions per image

Used for: object detection, segmentation, keypoints, and image captioning

COCO provides rich, diverse scenes—ideal for teaching models to understand real-world images.

🛠️ Workflow Summary
1️⃣ Dataset Preparation

Images resized to 299×299

Normalized to 0–1

Captions tokenized, encoded, padded

Vocabulary built using Keras Tokenizer

2️⃣ Feature Extraction (Encoder – CNN)

A pre-trained InceptionV3 extracts high-level visual features by removing the classification head, outputting compact image representations.

3️⃣ Caption Generation (Decoder – LSTM)

An LSTM network takes the image features and generates captions word-by-word, ensuring sentence flow and grammar.

4️⃣ Attention Mechanism

Allows the model to “look” at important regions while forming each word.

5️⃣ Training & Evaluation

Optimizer: Adam

Loss: Categorical Cross-Entropy

Split: 80% training / 20% validation

Evaluation:

BLEU Score

Human Inspection

6️⃣ Fine-Tuning

Unfreezing CNN layers → improves visual detail recognition

Data augmentation: rotation, zoom, flip

Leads to higher BLEU scores and better captions

🔍 Results
⭐ Strengths

Generates clear, grammatically correct, and relevant captions

Understands relationships (e.g., “A cat is sitting on the sofa.”)

Fine-tuned model performs significantly better

⚠️ Challenges

Struggles with abstract or artistic images

Misses subtle interactions (e.g., whispering, emotion cues)

Limited vocabulary for rare concepts not in COCO dataset

🧾 Code Structure
├── data/
│   ├── annotations.json
│   ├── images/
│   └── features/
├── src/
│   ├── extract_features.py
│   ├── preprocess_captions.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── inference.py
├── tokenizer.json
├── captions_data.npz
└── README.md

📘 Technologies Used

TensorFlow / Keras

NumPy, Pandas

InceptionV3 (ImageNet)

LSTM, Attention Mechanism

MS COCO Dataset

🌟 Conclusion

This project successfully builds a complete image-captioning system that bridges vision and language. It demonstrates how CNNs and LSTMs can work together to generate meaningful image descriptions.

The model works well for common, natural scenes and sets the stage for more advanced architectures.

🔮 Future Work

🚀 Integrate Vision Transformers (ViT)
🤝 Use multimodal transformer-based captioning
📈 Expand dataset for rare or abstract concepts
🗣️ Add multilingual caption generation

🙌 Team

Team-D — EvoAstra Major Project
