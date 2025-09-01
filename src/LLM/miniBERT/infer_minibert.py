import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

torch.set_default_dtype(torch.float64)

MAX_LEN    = 12
MODEL_DIR  = "bert_mini_sst2_finetuned"
PARAM_JSON = "bert_mini_params.json"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Running minibert inference...")

class BertMiniSST2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            MODEL_DIR, add_pooling_layer=False
        ).to(torch.double)
        self.classifier = nn.Linear(256, 2, dtype=torch.double)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)

with open(PARAM_JSON, "r") as f:
    param_dict = json.load(f)

model = BertMiniSST2().to(DEVICE)
state = model.state_dict()
for name, tensor in state.items():
    arr = np.array(param_dict[name], dtype=np.float64)
    state[name] = torch.from_numpy(arr).to(DEVICE)
model.load_state_dict(state)
print(f"Loaded all parameters from '{PARAM_JSON}'.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
sentence  = "The movie was delightful and I enjoyed every moment."
enc = tokenizer(
    sentence,
    truncation=True,
    padding='max_length',
    max_length=MAX_LEN,
    return_tensors='pt'
)
inputs = {
    "input_ids":      enc["input_ids"].to(DEVICE),
    "attention_mask": enc["attention_mask"].to(DEVICE),
}

activations = {}
def make_pre_hook(name):
    def hook(module, inp):
        activations[name] = inp[0].detach().cpu().tolist()
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.Embedding, nn.Linear, nn.LayerNorm)):
        module.register_forward_pre_hook(make_pre_hook(name))

emb_out    = {}
block0_out = {}
block1_out = {}
block2_out = {}
block3_out = {}
cls_out    = {}

def emb_forward_hook(module, inp, out):
    emb_out['layernorm'] = out.detach().cpu().tolist()

def block0_forward_hook(module, inp, out):
    block0_out['layernorm'] = out.detach().cpu().tolist()

def block1_forward_hook(module, inp, out):
    block1_out['layernorm'] = out.detach().cpu().tolist()

def block2_forward_hook(module, inp, out):
    block2_out['layernorm'] = out.detach().cpu().tolist()

def block3_forward_hook(module, inp, out):
    block3_out['layernorm'] = out.detach().cpu().tolist()

def cls_forward_hook(module, inp, out):
    cls_out['logits'] = out.detach().cpu().tolist()

model.bert.embeddings.LayerNorm.register_forward_hook(emb_forward_hook)
for i, blk in enumerate(model.bert.encoder.layer[:4]):
    blk.output.LayerNorm.register_forward_hook(
        [block0_forward_hook, block1_forward_hook, block2_forward_hook, block3_forward_hook][i]
    )
model.classifier.register_forward_hook(cls_forward_hook)

model.eval()
with torch.no_grad():
    logits = model(**inputs)

pred = int(logits.argmax(dim=1).item())
#print(f"Prediction: {pred}")

print("\n=== Embedding LayerNorm Done ===")
emb = np.array(emb_out['layernorm'], dtype=np.float64)
#print("Shape:", emb.shape)
#print(emb)

for idx, b_out in enumerate((block0_out, block1_out, block2_out, block3_out)):
    print(f"\n=== Encoder Block {idx} LayerNorm Done ===")
    arr = np.array(b_out['layernorm'], dtype=np.float64)
    #print("Shape:", arr.shape)
    #print(arr)

print("\n=== Classifier Done ===")
cl = np.array(cls_out['logits'], dtype=np.float64)
#print("Shape:", cl.shape)
#print(cl)

out_file = "bert_mini_inference_layer_inputs.json"
with open(out_file, "w") as f:
    json.dump(activations, f, indent=2)
print(f"\nSaved layer-by-layer inputs to '{out_file}'")
