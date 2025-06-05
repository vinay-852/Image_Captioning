import streamlit as st
import torch
from transformers import GPT2Tokenizer
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from src import ImageCaptioningModel, generate_caption

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = ImageCaptioningModel()
    model.text_decoder.resize_token_embeddings(len(tokenizer))

    checkpoint_path = hf_hub_download(
        repo_id="vinay-pepakayala/image_captioning",
        filename="image_captioning.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_tensor = image_transforms(image).unsqueeze(0).to(device)

    if st.button("Generate Caption"):
        with torch.no_grad():
            caption = generate_caption(model, image_tensor.squeeze(0).cpu(), tokenizer, device)
        st.success("Caption generated!")
        st.write(f"**Generated Caption:** {caption}")
        st.download_button(
            label="Download Caption",
            data=caption,
            file_name="caption.txt",
            mime="text/plain"
        )
