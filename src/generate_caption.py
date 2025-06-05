import torch

def generate_caption(model, image, tokenizer,device, max_length=20):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        vision_features = model.visual_encoder(image)
        vision_embeds = model.vision_proj(vision_features)
        vision_embeds_proj = model.image_proj_to_text(vision_embeds)

        gpt2_emb_dim = model.text_decoder.transformer.wte.weight.shape[1]
        vision_embeds_proj = vision_embeds_proj.view(1, model.n_prefix_tokens, gpt2_emb_dim)
        vision_embeds_proj = model.vision_transformer_encoder(vision_embeds_proj)

        generated_ids = model.text_decoder.generate(
            inputs_embeds=vision_embeds_proj,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)