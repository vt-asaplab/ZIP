import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()
torch.set_default_dtype(torch.float64)

BATCH_SIZE = 32
EPOCHS     = 2
LR         = 2e-5
MAX_LEN    = 64
MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F = 16  # fractional bits for fixed point
SCALE = 1 << F
INT_MIN, INT_MAX = -2**31, 2**31 - 1

def clamp_int32(x):
    return torch.clamp(x, INT_MIN, INT_MAX).to(torch.int32)

@torch.no_grad()
def fp_to_fxpt(x: torch.Tensor) -> torch.Tensor:
    q = torch.round(x * SCALE).to(torch.int64)
    q = torch.clamp(q, INT_MIN, INT_MAX).to(torch.int32)
    return q

@torch.no_grad()
def fxpt_to_fp(q: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float64) / SCALE

def fxp_add(a, b):
    return clamp_int32(a.to(torch.int64) + b.to(torch.int64))

def fxp_sub(a, b):
    return clamp_int32(a.to(torch.int64) - b.to(torch.int64))

def fxp_mul(a, b):
    prod = (a.to(torch.int64) * b.to(torch.int64)) >> F
    return clamp_int32(prod)

def fxp_matmul(a, w):
    out = torch.matmul(a.to(torch.int64), w.to(torch.int64).T) >> F
    return clamp_int32(out)

def fxp_div(a, b):
    num = (a.to(torch.int64) << F)
    res = num // b.to(torch.int64)
    return clamp_int32(res)

def fxp_rsqrt(x, iters=2):
    xf = x.to(torch.float64) / SCALE
    y0 = 1.0 / torch.sqrt(xf + 1e-12)
    y = fp_to_fxpt(y0)

    half = fp_to_fxpt(torch.tensor(0.5, dtype=torch.float64))
    three_half = fp_to_fxpt(torch.tensor(1.5, dtype=torch.float64))

    for _ in range(iters):
        y2   = fxp_mul(y, y)              
        xy2  = fxp_mul(x, y2)            
        term = fxp_sub(three_half, fxp_mul(half, xy2))
        y    = fxp_mul(y, term)
    return y

def fxp_gelu(x):
    c0 = fp_to_fxpt(torch.tensor(math.sqrt(2/math.pi)))
    c1 = fp_to_fxpt(torch.tensor(0.044715))
    half = fp_to_fxpt(torch.tensor(0.5))
    one  = fp_to_fxpt(torch.tensor(1.0))
    n27  = fp_to_fxpt(torch.tensor(27.0))
    n9   = fp_to_fxpt(torch.tensor(9.0))

    x3   = fxp_mul(fxp_mul(x, x), x)      # x^3
    inner= fxp_mul(c0, fxp_add(x, fxp_mul(c1, x3)))
    inner2= fxp_mul(inner, inner)
    num  = fxp_mul(inner, fxp_add(n27, inner2))
    den  = fxp_add(n27, fxp_mul(n9, inner2))
    t    = fxp_div(num, den)              # tanh approx
    return fxp_mul(fxp_mul(half, x), fxp_add(one, t))

LUT_SIZE = 256
exp2_lut = fp_to_fxpt(torch.pow(2.0, torch.linspace(-16, 0, LUT_SIZE, dtype=torch.float64)))

def fxp_exp2(z):
    zf = fxpt_to_fp(z)
    zf = torch.clamp(zf, -16.0, 0.0)
    pos = (zf + 16.0) * (LUT_SIZE-1) / 16.0
    idx0 = torch.clamp(pos.long(), 0, LUT_SIZE-1)
    return exp2_lut[idx0]

def fxp_softmax(logits):
    m = logits.max(dim=-1, keepdim=True).values
    l_shift = fxp_sub(logits, m)

    inv_ln2 = fp_to_fxpt(torch.tensor(1.0/math.log(2.0)))
    z = fxp_mul(l_shift, inv_ln2) 

    e = fxp_exp2(z)        
    sum_e = torch.sum(e.to(torch.int64), dim=-1, keepdim=True)
    return fxp_div(e, sum_e)

def fxp_layernorm(x, gamma, beta, eps=1e-12):
    dim = x.shape[-1]
    dim_inv = fp_to_fxpt(torch.tensor(1.0/dim, dtype=torch.float64))

    mean = torch.sum(x.to(torch.int64), dim=-1, keepdim=True) 
    mean = (mean * dim_inv.to(torch.int64)) >> F
    mean = clamp_int32(mean)

    xc   = fxp_sub(x, mean)
    var  = torch.sum((xc.to(torch.int64)*xc.to(torch.int64)) >> F, dim=-1, keepdim=True)
    var  = (var * dim_inv.to(torch.int64)) >> F
    var  = clamp_int32(var)

    var_eps = fxp_add(var, fp_to_fxpt(torch.tensor(eps)))
    inv_std = fxp_rsqrt(var_eps)

    norm = fxp_mul(xc, inv_std)
    out  = fxp_add(fxp_mul(norm, gamma), beta)
    return out

class FXPLinear(nn.Module):
    def __init__(self, w_fp, b_fp):
        super().__init__()
        self.w = fp_to_fxpt(w_fp) 
        self.b = fp_to_fxpt(b_fp) if b_fp is not None else None

    def forward(self, x):
        # x: [B, in]
        y = fxp_matmul(x, self.w)
        if self.b is not None:
            y = fxp_add(y, self.b)
        return y

class FXPBertLayer(nn.Module):
    def __init__(self, src, hidden_size=256, num_heads=4):
        super().__init__()
        self.h = num_heads
        self.d = hidden_size // num_heads

        # QKV
        self.q = FXPLinear(src.attention.self.query.weight,  src.attention.self.query.bias)
        self.k = FXPLinear(src.attention.self.key.weight,    src.attention.self.key.bias)
        self.v = FXPLinear(src.attention.self.value.weight,  src.attention.self.value.bias)

        self.attn_out = FXPLinear(src.attention.output.dense.weight,
                                  src.attention.output.dense.bias)

        self.ln1_g = fp_to_fxpt(src.attention.output.LayerNorm.weight)
        self.ln1_b = fp_to_fxpt(src.attention.output.LayerNorm.bias)

        # FFN
        self.fc1 = FXPLinear(src.intermediate.dense.weight, src.intermediate.dense.bias)
        self.fc2 = FXPLinear(src.output.dense.weight,       src.output.dense.bias)

        self.ln2_g = fp_to_fxpt(src.output.LayerNorm.weight)
        self.ln2_b = fp_to_fxpt(src.output.LayerNorm.bias)

        self.scale_qk = fp_to_fxpt(torch.tensor(1.0 / math.sqrt(self.d)))

    def forward(self, x, attn_mask):
        B, S, H = x.shape

        # --- Self-attention ---
        q = self.q(x.view(-1,H)).view(B,S,H)
        k = self.k(x.view(-1,H)).view(B,S,H)
        v = self.v(x.view(-1,H)).view(B,S,H)

        # reshape heads
        q = q.view(B, S, self.h, self.d).permute(0,2,1,3)   # B,h,S,d
        k = k.view(B, S, self.h, self.d).permute(0,2,1,3)
        v = v.view(B, S, self.h, self.d).permute(0,2,1,3)

        # attn scores = q @ k^T * scale_qk
        scores = torch.zeros(B,self.h,S,S,dtype=torch.int32,device=q.device)
        for b in range(B):
            for h in range(self.h):
                a = q[b,h].to(torch.int64) 
                btk= k[b,h].to(torch.int64).transpose(0,1) 
                sc = (a @ btk) >> F         
                sc = clamp_int32(sc)
                scores[b,h] = fxp_mul(sc, self.scale_qk)

        scores = scores + attn_mask
        attn = fxp_softmax(scores)

        ctx = torch.zeros(B,self.h,S,self.d,dtype=torch.int32,device=v.device)
        for b in range(B):
            for h in range(self.h):
                a = attn[b,h].to(torch.int64)   
                vv= v[b,h].to(torch.int64)     
                c = (a @ vv) >> F
                ctx[b,h] = clamp_int32(c)
        ctx = ctx.permute(0,2,1,3).contiguous().view(B,S,H)

        attn_out = self.attn_out(ctx.view(-1,H)).view(B,S,H)
        # residual + LN1
        x = fxp_add(x, attn_out)
        x = fxp_layernorm(x, self.ln1_g, self.ln1_b)

        # FFN
        h1 = self.fc1(x.view(-1,H)).view(B,S,-1)
        h1 = fxp_gelu(h1)
        h2 = self.fc2(h1.view(-1,h1.size(-1))).view(B,S,H)

        x = fxp_add(x, h2)
        x = fxp_layernorm(x, self.ln2_g, self.ln2_b)
        return x

class FXPBert(nn.Module):
    def __init__(self, fp_model):
        super().__init__()
        # Embeddings
        emb = fp_model.bert.embeddings
        self.word = fp_to_fxpt(emb.word_embeddings.weight)
        self.pos  = fp_to_fxpt(emb.position_embeddings.weight)
        self.tok  = fp_to_fxpt(emb.token_type_embeddings.weight)
        self.ln_g = fp_to_fxpt(emb.LayerNorm.weight)
        self.ln_b = fp_to_fxpt(emb.LayerNorm.bias)

        # Encoder layers
        self.layers = nn.ModuleList([
            FXPBertLayer(l) for l in fp_model.bert.encoder.layer
        ])

    def forward(self, input_ids, token_type_ids, attention_mask):
        B,S = input_ids.shape
        we = self.word[input_ids]    
        pe = self.pos[torch.arange(S,device=we.device)].unsqueeze(0).expand(B,-1,-1)
        te = self.tok[token_type_ids]
        x  = fxp_add(fxp_add(we, pe), te)

        x = fxp_layernorm(x, self.ln_g, self.ln_b)

        mask = (1 - attention_mask).unsqueeze(1).unsqueeze(2)  
        big_neg = fp_to_fxpt(torch.tensor(-1e9))
        attn_mask_fx = mask.to(torch.int32) * big_neg

        for layer in self.layers:
            x = layer(x, attn_mask_fx)

        return x

class FXPBertMiniSST2(nn.Module):
    def __init__(self, fp_model):
        super().__init__()
        self.backbone = FXPBert(fp_model)
        cls_w = fp_model.classifier.weight
        cls_b = fp_model.classifier.bias
        self.classifier = FXPLinear(cls_w, cls_b)

    def forward(self, input_ids, attention_mask):
        token_type_ids = torch.zeros_like(input_ids)
        hs = self.backbone(input_ids, token_type_ids, attention_mask)
        cls_tok = hs[:,0]  # [B,H]
        logits = self.classifier(cls_tok)
        return logits

class BertMiniSST2_FP64(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False).to(torch.double)
        self.classifier = nn.Linear(256, 2, dtype=torch.double)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]
        return self.classifier(cls)

def train_epoch(model, loader, criterion, optimizer, ep):
    model.train()
    for step, batch in enumerate(loader,1):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss   = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step%200==0 or step==len(loader):
            print(f"Epoch {ep} | Step {step}/{len(loader)} | Loss {loss.item():.4f}")

@torch.no_grad()
def eval_fp(model, loader, tag):
    model.eval()
    correct=total=0
    for batch in loader:
        batch={k:v.to(DEVICE) for k,v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds  = logits.argmax(-1)
        correct += (preds==batch["label"]).sum().item()
        total   += preds.size(0)
    acc = 100*correct/total
    print(f"{tag} acc: {acc:.2f}%")
    return acc

@torch.no_grad()
def eval_fxp(model_fx, loader, tag):
    model_fx.eval()
    correct=total=0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        # run fxp
        logits = model_fx(input_ids, attn_mask)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds.cpu() == batch["label"]).sum().item()
        total   += preds.size(0)
    acc = 100*correct/total
    print(f"{tag} acc: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    raw_ds = load_dataset("glue","sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN)
    ds = raw_ds.map(tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ds["validation"], batch_size=BATCH_SIZE)

    # Train double precision
    fp_model  = BertMiniSST2_FP64().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(fp_model.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        train_epoch(fp_model, train_loader, criterion, optimizer, ep)
        eval_fp(fp_model, val_loader, f"ZIP (IEEE-754 double-precision) after epoch {ep}")

    fp_acc = eval_fp(fp_model, val_loader, "Final FP64")

    # Build fixed point model
    print("\nConverting to strict fixed‑point (F=16)…")
    fxp_model = FXPBertMiniSST2(fp_model).to(DEVICE)

    fxp_acc = eval_fxp(fxp_model, val_loader, "Baseline (w/FP) (fixed-point)")

    print("\nSummary:")
    print(f"ZIP (IEEE-754 double-precision) acc : {fp_acc:.2f}%")
    print(f"Baseline (w/FP) (fixed-point) acc : {fxp_acc:.2f}% (drop {fp_acc-fxp_acc:.2f} pp)")
 