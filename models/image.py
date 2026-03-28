import torch
import torchvision
import torch.nn as nn
from einops import rearrange

class ImageEncoder_pool(nn.Module):
    def __init__(self, args, configs):
        super(ImageEncoder_pool, self).__init__()
        self.args = args
        self.configs = configs
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.pool_func = (
            nn.AdaptiveMaxPool2d
            if args.img_embed_pool_type == 'max'
            else nn.AdaptiveAvgPool2d)

    def forward(self, x):
        out = self.model(x)
        model_out = out.size()[-2:]
        W = int(model_out[0] / 2)
        H = int(model_out[1] / 2)
        pool = self.pool_func((W, H))
        out = torch.flatten(pool(out), start_dim=2).transpose(1, 2).contiguous() 
        # random pixel sampling
        random_sampling = torch.randperm(out.size(1))[:self.configs['num_image_embeds']]
        random_sampling, _ = torch.sort(random_sampling)
        random_sample = out[:, random_sampling]
        return random_sample


from open_clip import create_model_from_pretrained
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
# resnet50 + text emb -> fusion ->  bert based 
class BioMedCLIPImageEncoder(nn.Module):
    def __init__(self, args, configs):
        super().__init__()
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BioMedCLIP
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, self.preprocess = create_model_from_pretrained(model_name)

        # Extract vision encoder
        self.encoder = model.visual.trunk

        # STEP 1: freeze all except last 2 trans blocks + norm
        for name, p in self.encoder.named_parameters():
            # Unfreeze last 2 transformer blocks + LayerNorm
            if "blocks.10" in name or "blocks.11" in name or "norm" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # STEP 2: Add LoRA to attention layers
        lora_config = LoraConfig(
            r=8,                     # rank (try 4, 8, 16)
            lora_alpha=16,
            target_modules=["qkv", "proj"],  # VERY IMPORTANT for ViT
            lora_dropout=0.1,
            bias="none"
        )

        self.encoder = get_peft_model(self.encoder, lora_config)

        # Print trainable params (sanity check)
        self.encoder.print_trainable_parameters()

        self.encoder.to(self.device)

        # STEP 3: Get embedding dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            out = self.encoder.forward_features(dummy)

            if isinstance(out, dict):
                out = out["last_hidden_state"]

            visual_dim = out.shape[-1]

        # Projection (MedViLL expects 768)
        self.proj = nn.Identity() if visual_dim == 768 else nn.Linear(visual_dim, 768)
        self.proj.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        out = self.encoder.forward_features(x)
        if isinstance(out, dict):
            out = out["last_hidden_state"]

        # Remove CLS token → B x 196 x 768
        out = out[:, 1:, :]

        # Token sampling: 196 → 64 evenly spaced
        num_tokens = out.size(1)                          # 196
        num_select = self.configs['num_image_embeds']     # 64
        
        if num_tokens > num_select:
            step = num_tokens // num_select               # 196 // 64 = 3
            idx = torch.arange(0, num_tokens, step, device=out.device)[:num_select]
            out = out[:, idx, :]                          # B x 64 x 768
        # if num_tokens <= num_select, keep all (no sampling needed)

        # Project to 768 if needed
        out = self.proj(out)                              # B x 64 x 768

        # Positional embeddings
        vis_pe = torch.arange(out.size(1), device=out.device)
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), -1)

        return out, vis_pe
# class BioMedCLIPImageEncoder(nn.Module):
#     def __init__(self, args, configs):
#         super().__init__()
#         self.configs = configs
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load model
#         model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#         model, self.preprocess = create_model_from_pretrained(model_name)

#         # 🔥 Extract ONLY vision transformer
#         self.encoder = model.visual.trunk
#         self.encoder.to(self.device)
#         self.encoder.eval()

#         # Freeze
#         for p in self.encoder.parameters():
#             p.requires_grad = False

#         # --- Get embedding dim ---
#         with torch.no_grad():
#             dummy = torch.randn(1, 3, 224, 224).to(self.device)
#             out = self.encoder(dummy)

#             if isinstance(out, dict):
#                 out = out["last_hidden_state"]

#             visual_dim = out.shape[-1]  # should be 768

#         # Projection (keep 768 for MedViLL)
#         self.proj = nn.Identity() if visual_dim == 768 else nn.Linear(visual_dim, 768)

#     def forward(self, x):
#         x = x.to(self.device)

#         # with torch.no_grad():
#         out = self.encoder.forward_features(x)

#         if isinstance(out, dict):
#             out = out["last_hidden_state"]

#         #  remove CLS token
#         out = out[:, 1:, :]   # B x 196 x 768

#         # sample tokens (match MedViLL)
#         num_tokens = out.size(1)
#         num_select = self.configs['num_image_embeds']

#         idx = torch.randperm(num_tokens, device=out.device)[:num_select]
#         idx, _ = torch.sort(idx)

#         out = out[:, idx, :]  # B x N x 768

#         # Projection (usually identity)
#         out = self.proj(out)

#         # Positional embeddings
#         vis_pe = torch.arange(out.size(1), device=out.device)
#         vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), -1)

#         return out, vis_pe

class ImageEncoder_cnn(nn.Module):
    def __init__(self, args, configs):
        super(ImageEncoder_cnn, self).__init__()
        self.args = args
        self.configs = configs
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)  # 512x512: torch.Size([16, 2048, 16, 16])
        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()
        
        vis_pe = torch.arange(out.size(1), dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), out.size(1))

        random_sampling = torch.randperm(out.size(1))[:self.configs['num_image_embeds']]
        random_sampling, _ = torch.sort(random_sampling)

        random_sample = out[:, random_sampling]
        random_position = vis_pe[:, random_sampling]
        return random_sample, random_position

import torch
import torch.nn as nn
from transformers import AutoModel

class BioViLTImageEncoder(nn.Module):
    def __init__(self, args, configs):
        super().__init__()
        self.args    = args
        self.configs = configs

        # Load full BioViL-T VLP model
        full_model = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            trust_remote_code=True
        )

        # From model.py we can see the full model has:
        #   full_model.image_model          → ImageModel instance
        #   full_model.image_model.encoder  → ResNet backbone (get_encoder_from_type)
        #   full_model.image_model.projector→ MLP projection head
        # We extract ONLY the encoder backbone, discard everything else
        self.encoder = full_model.base_model.encoder
        del full_model  # free VRAM — drop text tower + projector

        # Verify output dim via dummy forward
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            patch_x, pooled_x = self.encoder(dummy, return_patch_embeddings=True)
            # patch_x:  B x C x H x W  (spatial feature map)
            # pooled_x: B x C          (global average pooled)
            print(f"[BioViL-T] encoder patch output: {patch_x.shape}")
            print(f"[BioViL-T] encoder pooled output: {pooled_x.shape}")
            visual_dim = patch_x.shape[1]  # C — typically 2048 for ResNet-50

        # Project to 2048 if not already (ResNet-50 is already 2048)
        # self.proj = nn.Linear(visual_dim, 2048) if visual_dim != 2048 else nn.Identity()
        self.proj = nn.Linear(visual_dim, 768)
        # Freeze all encoder layers, unfreeze from layer4 onward
        # Mirrors original MedViLL strategy for CNN encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        for child in list(self.encoder.children())[5:]:
            for p in child.parameters():
                p.requires_grad = True

    def forward(self, x):
        # Use the exact same API shown in model.py ImageModel.forward():
        #   patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        # patch_x: B x 2048 x H x W  (e.g. 7x7 for 224px, 16x16 for 512px)

        # Flatten spatial dims → token sequence
        # Identical to ImageEncoder_cnn forward()
        out = torch.flatten(patch_x, start_dim=2).transpose(1, 2).contiguous()
        out = self.proj(out)
        # out: B x (H*W) x 2048

        vis_pe = torch.arange(out.size(1), dtype=torch.long, device=x.device)
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), out.size(1))

        # Random spatial sampling to num_image_embeds — same as ImageEncoder_cnn
        random_sampling = torch.randperm(out.size(1))[:self.configs['num_image_embeds']]
        random_sampling, _ = torch.sort(random_sampling)

        return out[:, random_sampling], vis_pe[:, random_sampling]

class fully_use_cnn(nn.Module):
    def __init__(self):
        super(fully_use_cnn, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        pool_func = (nn.AdaptiveAvgPool2d)
        self.pool = pool_func((3, 1))

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()
        vis_pe = torch.arange(out.size()[1], dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size()[0], out.size()[1])
        return out, vis_pe


class Img_patch_embedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

    def forward(self, img, mask=None):
        img_size = img.size()
        p = self.patch_size
        out = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        out = self.patch_to_embedding(out)
        return out
