import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)

def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    elif name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation '{name}'. Choose from: gelu, elu, selu.")

class LeNetAct(nn.Module):
    def __init__(self, act_name: str):
        super().__init__()
        act = lambda: make_activation(act_name)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.act1  = act()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2  = act()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.act3  = act()
        self.fc2   = nn.Linear(120, 84)
        self.act4  = act()
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)

def main():
    p = argparse.ArgumentParser(description="Inference + layer-input dump for LeNet on MNIST (fp64)")
    p.add_argument("--act", type=str, default="elu", choices=["gelu", "elu", "selu"],
                   help="activation to use: gelu (tanh approx), elu, selu")
    p.add_argument("--params", type=str, default=None,
                   help="path to params JSON (default: lenet_model_params.json, or lenet_params.json fallback)")
    p.add_argument("--out", type=str, default=None,
                   help="output JSON for layer inputs (default: lenet_inference_layer_inputs.json)")
    p.add_argument("--batch_size", type=int, default=1, help="MNIST test batch size")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = p.parse_args()

    act = args.act.lower()
    params_path = args.params or f"lenet_model_params.json"
    out_path = args.out or f"lenet_inference_layer_inputs.json"

    if not (torch.ops._C and torch.jit) and False: 
        pass

    try:
        with open(params_path, "r") as f:
            param_dict = json.load(f)
    except FileNotFoundError:
        if args.params is None and params_path != "lenet_model_params.json":
            try:
                with open("lenet_model_params.json", "r") as f:
                    param_dict = json.load(f)
                print(f"[warn] {params_path} not found. Loaded fallback 'lenet_model_params.json'.")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find params JSON: '{params_path}' (or 'lenet_model_params.json')."
                )
        else:
            raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNetAct(act).to(device).double()

    state = model.state_dict()
    for name in list(state.keys()):
        if name not in param_dict:
            raise KeyError(f"Missing key in params JSON: {name}")
        arr = np.array(param_dict[name], dtype=np.float64)
        state[name] = torch.from_numpy(arr).to(device)
    model.load_state_dict(state)
    print(f"Loaded parameters from '{params_path}'.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    image, label = next(iter(test_loader))
    image = image.to(device).double()

    activations = {}

    def pre_hook(name):
        def _h(module, inp):
            activations[name] = inp[0].detach().cpu().numpy().tolist()
        return _h

    model.conv1.register_forward_pre_hook(pre_hook("conv1_in"))
    model.act1 .register_forward_pre_hook(pre_hook("act1_in"))
    model.pool1.register_forward_pre_hook(pre_hook("pool1_in"))

    model.conv2.register_forward_pre_hook(pre_hook("conv2_in"))
    model.act2 .register_forward_pre_hook(pre_hook("act2_in"))
    model.pool2.register_forward_pre_hook(pre_hook("pool2_in"))

    model.fc1  .register_forward_pre_hook(pre_hook("fc1_in"))
    model.act3 .register_forward_pre_hook(pre_hook("act3_in"))

    model.fc2  .register_forward_pre_hook(pre_hook("fc2_in"))
    model.act4 .register_forward_pre_hook(pre_hook("act4_in"))

    model.fc3  .register_forward_pre_hook(pre_hook("fc3_in"))

    model.eval()
    with torch.no_grad():
        out = model(image)

    pred = out.argmax(dim=1).item()

    meta = {
        "_meta": {
            "activation": act,
            "dtype": "float64",
            "pred": int(pred),
            "label": int(label.item())
        }
    }
    meta.update(activations)

    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved layer-by-layer inputs to '{out_path}'.")

if __name__ == "__main__":
    main()
