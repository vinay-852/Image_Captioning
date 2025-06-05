import os
import torch
from transformers import GPT2Tokenizer
from PIL import Image
from .src import ImageCaptioningModel, generate_caption
from torchvision import transforms

device = 'mps' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = ImageCaptioningModel().to(device)
model.text_decoder.resize_token_embeddings(len(tokenizer))

checkpoint = torch.load('checkpoints/epoch_3.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "path_to_your_image.jpg"  
image = Image.open(image_path).convert("RGB")
image = image_transforms(image).unsqueeze(0).to(device)  

with torch.no_grad():
    generated_caption = generate_caption(model, image.squeeze(0).cpu(), tokenizer, device)

print(f"Generated Caption: {generated_caption}")
