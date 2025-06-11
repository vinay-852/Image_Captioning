# Image Captioning Model

This project implements an **Image Captioning** system that generates natural language descriptions for images, leveraging a **Vision Transformer (ViT)** as the encoder and **GPT-2** as the decoder.

---

## Key Features

Uses **ViT (vit\_base\_patch16\_224)** for robust visual feature extraction

Uses **GPT-2** for rich, coherent language generation

Custom projection and transformer modules to bridge image and text domains

Ready-to-run **Streamlit** app for interactive demos

Modular and customizable architecture

---

## Installation

1 **Clone the repository:**

```bash
git clone https://github.com/vinay-852/Image_Captioning.git
cd Image_Captioning
```

2 **Install dependencies:**

```bash
pip install -r requirements.txt
```

3 **Launch the Streamlit app:**

```bash
streamlit run app.py
```

---

## Dataset

The model is trained on the **Flickr31k** dataset, known for its diverse and descriptive image captions.

---

## Model Architecture

The architecture consists of the following components:

### 1 **Image Encoder (ViT)**

* A pretrained Vision Transformer (`vit_base_patch16_224`) from the `timm` library.
* Extracts high-level visual features from input images.
* The classification head is removed (`nn.Identity()`) since we only need feature representations.
* The encoder parameters are **frozen** to leverage pretrained knowledge without updating them during training.

### 2 **Visual Feature Projection**

* The ViT’s output is mapped to a common **embedding dimension** (`embed_dim=512`) using a linear layer.
* A **deep projection head** reshapes these embeddings into a sequence of **prefix tokens** that can be interpreted by GPT-2.
* Output: `[batch_size, n_prefix_tokens, gpt2_emb_dim]`, where `n_prefix_tokens=5` by default.

### 3 **Vision Transformer Encoder (Optional)**

* A 2-layer **Transformer Encoder** (`nn.TransformerEncoder`) refines the visual prefix tokens.
* It allows **self-attention** across image patches, capturing complex object relationships and spatial context.

### 4 **Language Decoder (GPT-2)**

* GPT-2 is used as a **language model** to generate the final captions.
* Text input tokens are converted to embeddings (`text_embeds`) using GPT-2’s embedding layer.
* The **visual prefixes** are concatenated to the **beginning** of the text embeddings.
* This makes GPT-2 treat the visual information as context while generating text.

### 5 **Loss Masking**

* During training, visual prefix tokens are **masked out** of the loss calculation (using `-100`).
* Only the **text portion** of the sequence contributes to the captioning loss, ensuring accurate learning.

---

## Common Doubts

 **Why use GPT-2?**
GPT-2’s generative capability allows it to produce coherent, contextually rich captions.

 **Why freeze ViT?**
ViT already learns generic visual features from large-scale data (like ImageNet). Freezing avoids overfitting and reduces memory usage.

 **Role of Vision Transformer Encoder?**
It helps **refine the image’s representation**, letting different image patches “talk” to each other before caption generation.

 **What are prefix tokens?**
They encode the image’s visual context as “virtual words” that GPT-2 sees before actual text.

 **Can I change the number of prefix tokens?**
Yes! Increase or decrease `n_prefix_tokens` depending on how much visual context you want GPT-2 to use.

---

## 🔧 Training & Customization

* Train the model using **cross-entropy loss** on your dataset of images and captions.
* Adjust hyperparameters:

  * `embed_dim` for embedding size
  * `n_prefix_tokens` for visual context
  * Transformer encoder layers for refinement
* Explore **cross-attention** methods (like BLIP or Flamingo) for richer integration of vision and language.

---

## 📈 Results & Usage

* At inference time, the model generates a natural language caption **conditioned on the image’s visual representation**.
* Ideal for applications like:

  * Automated image annotation
  * Visual storytelling
  * Accessibility (descriptive captions for the visually impaired)

---

## Contribution & Feedback

Feel free to fork, contribute, or suggest improvements!
If you have questions, open an issue or contact me directly.

---
