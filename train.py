import os
import torch
from transformers import GPT2Tokenizer
from .src import ImageCaptioningModel, generate_caption, load_data, FlickrImageCaptionDataset
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms


dataset = load_data()
image_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

flickrdataset = FlickrImageCaptionDataset(dataset, image_transforms)
train_dataloader = DataLoader(flickrdataset, batch_size=96, shuffle=True)
num_batches = len(train_dataloader)
print(num_batches)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = ImageCaptioningModel().to(device)
model.text_decoder.resize_token_embeddings(len(tokenizer))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

epochs = 3
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, captions) in enumerate(train_dataloader):
        images = images.to(device)
        encodings = tokenizer(captions, return_tensors='pt', padding=True, truncation=True, max_length=50)
        input_ids = encodings['input_ids'].to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images, input_ids, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        print(f"Batch {batch_idx+1} | Loss: {loss.item():.4f}")

        if batch_idx % 20 == 0:
            model.eval()
            with torch.no_grad():
                sample_image = images[0].cpu()
                generated_caption = generate_caption(model, sample_image, tokenizer, device)
                print(f"Generated: {generated_caption}")
                print(f"Expected: {captions[0]}")
            model.train()
    scheduler.step()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint_path = f"checkpoints/epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")