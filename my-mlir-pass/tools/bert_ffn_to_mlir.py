from pathlib import Path

import torch
from torch_mlir import fx

program = torch.export.load("build/bert-tiny-layer0-ffn.pt2")
module = fx.export_and_import(program, output_type="linalg-on-tensors")
Path("build/bert-tiny-layer0-ffn.mlir").write_text(str(module))
