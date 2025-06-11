import torch
import torch.nn as nn
import timm
from transformers import GPT2LMHeadModel


class ImageCaptioningModel(nn.Module):
    def __init__(self, image_model_name='vit_base_patch16_224', text_model_name='gpt2', embed_dim=512, n_prefix_tokens=5):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens

        self.visual_encoder = timm.create_model(image_model_name, pretrained=True)
        self.visual_encoder.head = nn.Identity()
        vision_output_dim = self.visual_encoder.num_features
        self.vision_proj = nn.Linear(vision_output_dim, embed_dim)

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        self.text_decoder = GPT2LMHeadModel.from_pretrained(text_model_name)
        gpt2_emb_dim = self.text_decoder.transformer.wte.weight.shape[1]

        self.image_proj_to_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_prefix_tokens * gpt2_emb_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gpt2_emb_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.vision_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, images, input_ids, labels=None):
        vision_features = self.visual_encoder(images)
        vision_embeds = self.vision_proj(vision_features)
        vision_embeds_proj = self.image_proj_to_text(vision_embeds)

        B = vision_embeds_proj.shape[0]
        gpt2_emb_dim = self.text_decoder.transformer.wte.weight.shape[1]
        vision_embeds_proj = vision_embeds_proj.view(B, self.n_prefix_tokens, gpt2_emb_dim)

        vision_embeds_proj = self.vision_transformer_encoder(vision_embeds_proj)

        text_embeds = self.text_decoder.transformer.wte(input_ids) #word token embeddings
        gpt2_inputs_embeds = torch.cat([vision_embeds_proj, text_embeds], dim=1)

        if labels is not None:
            labels = torch.cat([
                torch.full((B, self.n_prefix_tokens), -100, device=labels.device),
                labels
            ], dim=1)

        outputs = self.text_decoder(
            inputs_embeds=gpt2_inputs_embeds,
            attention_mask=None,
            labels=labels
        )
        return outputs