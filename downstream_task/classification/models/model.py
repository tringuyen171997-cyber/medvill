# import torch
# import torch.nn as nn
# from transformers import BertConfig, BertModel
# from models.image import ImageEncoder

# class ImageBertEmbeddings(nn.Module):
#     def __init__(self, args, embeddings):
#         super(ImageBertEmbeddings, self).__init__()
#         self.args = args
#         self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
#         self.position_embeddings = embeddings.position_embeddings
#         self.token_type_embeddings = embeddings.token_type_embeddings
#         self.word_embeddings = embeddings.word_embeddings
#         self.LayerNorm = embeddings.LayerNorm
#         self.dropout = nn.Dropout(p=args.dropout)

#     def forward(self, input_imgs, token_type_ids):
#         bsz = input_imgs.size(0)
#         seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

#         cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
#         cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
#         cls_token_embeds = self.word_embeddings(cls_id)

#         sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
#         sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
#         sep_token_embeds = self.word_embeddings(sep_id)

#         imgs_embeddings = self.img_embeddings(input_imgs)
#         token_embeddings = torch.cat(
#             [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1)

#         position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
#         position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
#         position_embeddings = self.position_embeddings(position_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#         embeddings = token_embeddings + position_embeddings + token_type_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


# class MultimodalBertEncoder(nn.Module):
#     def __init__(self, args):
#         super(MultimodalBertEncoder, self).__init__()
#         self.args = args

#         if args.init_model == "bert-base-scratch":
#             config = BertConfig.from_pretrained("bert-base-uncased")
#             bert = BertModel(config)
#         else:
#             bert = BertModel.from_pretrained(args.init_model)
#         self.txt_embeddings = bert.embeddings
#         self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
#         self.img_encoder = ImageEncoder(args)
#         self.encoder = bert.encoder
#         self.pooler = bert.pooler
#         self.clf = nn.Linear(args.hidden_sz, args.n_classes)

#     def forward(self, input_txt, attention_mask, segment, input_img):
#         bsz = input_txt.size(0)
#         attention_mask = torch.cat(
#             [
#                 torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
#                 attention_mask,
#             ],
#             dim=1)
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

#         try:
#             extended_attention_mask = extended_attention_mask.to(
#                 dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         except StopIteration:
#             extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         img_tok = (
#             torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
#             .fill_(0)
#             .cuda())
#         img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048

        
#         img_embed_out = self.img_embeddings(img, img_tok)
#         txt_embed_out = self.txt_embeddings(input_txt, segment)
#         encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
#         encoded_layers = self.encoder(encoder_input, extended_attention_mask)
#         return self.pooler(encoded_layers[-1])


# class MultimodalBertClf(nn.Module):
#     def __init__(self, args):
#         super(MultimodalBertClf, self).__init__()
#         self.args = args
#         self.enc = MultimodalBertEncoder(args)
#         self.clf = nn.Linear(args.hidden_sz, args.n_classes)

#     def forward(self, txt, mask, segment, img):
#         x = self.enc(txt, mask, segment, img)
#         return self.clf(x)
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoModel
from models.image import BioMedCLIPImageEncoder


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super().__init__()
        self.args = args
        # BioMedCLIP already outputs 768 — no extra projection needed
        self.img_embeddings     = nn.Linear(args.hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings    = embeddings.word_embeddings
        self.LayerNorm          = embeddings.LayerNorm
        self.dropout            = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        # input_imgs: B x num_image_embeds x 768
        bsz        = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # CLS + patches + SEP

        cls_id = torch.LongTensor(
            [self.args.vocab.stoi["[CLS]"]]).to(input_imgs.device)
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor(
            [self.args.vocab.stoi["[SEP]"]]).to(input_imgs.device)
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings  = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1)

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_imgs.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)

        position_embeddings    = self.position_embeddings(position_ids)
        token_type_embeddings  = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ── BERT text encoder ──────────────────────────────────────
        biomed_bert = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        if args.init_model == "bert-base-scratch":
            config = BertConfig.from_pretrained(biomed_bert)
            bert   = BertModel(config)
        else:
            # Load pretrained BiomedBERT
            bert = AutoModel.from_pretrained(biomed_bert)
            print(f"[MODEL] Loaded pretrained BERT: {biomed_bert}")

        self.txt_embeddings = bert.embeddings
        self.encoder        = bert.encoder
        self.pooler         = bert.pooler

        # ── Image encoder ─────────────────────────────────────────
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder    = BioMedCLIPImageEncoder(args)

        # ── Classification head ───────────────────────────────────
        self.clf = nn.Sequential(
            nn.Linear(args.hidden_sz, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, args.n_classes)
        )

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)

        # Encode image → B x num_image_embeds x 768
        img_features, _ = self.img_encoder(input_img)

        # Image token type ids (all 0)
        img_tok = torch.zeros(
            bsz, self.args.num_image_embeds + 2,
            dtype=torch.long, device=input_txt.device)

        # Extend attention mask for image tokens
        attention_mask = torch.cat([
            torch.ones(bsz, self.args.num_image_embeds + 2,
                       dtype=torch.long, device=input_txt.device),
            attention_mask
        ], dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Embeddings
        img_embed_out = self.img_embeddings(img_features, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)

        # Concatenate image + text
        encoder_input  = torch.cat([img_embed_out, txt_embed_out], dim=1)
        encoded_layers = self.encoder(
            encoder_input,
            attention_mask=extended_attention_mask
        )

        # Use last hidden state CLS token
        cls_output = encoded_layers.last_hidden_state[:, 0, :]
        return cls_output


class MultimodalBertClf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc  = MultimodalBertEncoder(args)
        self.clf  = nn.Sequential(
            nn.Linear(args.hidden_sz, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, args.n_classes)
        )

    def forward(self, txt, mask, segment, img):
        cls_output = self.enc(txt, mask, segment, img)
        return self.clf(cls_output)