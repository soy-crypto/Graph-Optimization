from pathlib import Path
import numpy as np

build = Path("build")
with np.load(build / "bert-tiny-layer0-ffn.npz") as saved:
    x = np.ascontiguousarray(saved["X"], dtype="<f4")
assert x.shape == (1, 64, 128)

wrapper = """
  func.func private @printF32(f32)
  func.func private @printNewline()

  func.func @check_ffn() {
    %x = arith.constant dense<"0xHEX_INPUT"> : tensor<1x64x128xf32>
    %y = func.call @main(%x) : (tensor<1x64x128xf32>) -> tensor<1x64x128xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    scf.for %token = %c0 to %c64 step %c1 {
      scf.for %feature = %c0 to %c128 step %c1 {
        %value = tensor.extract %y[%c0, %token, %feature] : tensor<1x64x128xf32>
        func.call @printF32(%value) : (f32) -> ()
        func.call @printNewline() : () -> ()
      }
    }
    return
  }
""".replace("HEX_INPUT", x.tobytes().hex())

for variant, name in [
    ("original", "bert-tiny-layer0-ffn.mlir"),
    ("upstream", "bert-tiny-layer0-ffn-upstream.mlir"),
]:
    source = (build / name).read_text()
    entry = "func.func @main("
    assert source.count(entry) == 1
    (build / f"ffn-{variant}-check.mlir").write_text(
        source.replace(entry, wrapper + "\n  " + entry, 1)
    )