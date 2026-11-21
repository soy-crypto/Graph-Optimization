"""Benchmark the two BERT-Tiny layer 0 FFN variants from the project root."""
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile

import numpy as np


BUILD = Path("build").resolve()
LLVM = Path(os.environ.get("MLIR_BUILD", Path.home() / "llvm-project/build")).resolve()
SOURCES = {
    "original": BUILD / "bert-tiny-layer0-ffn.mlir",
    "upstream": BUILD / "bert-tiny-layer0-ffn-upstream.mlir",
}
SHAPE = (1, 64, 128)

# This native timing loop is compiled separately from the FFN. Each indirect call
# runs the kernel on the supplied input; LLVM cannot fold the repeated calls.
DRIVER = r"""
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { tokens = 64, features = 128, count = tokens * features, repeats = 100 };

typedef struct {
    float *allocated, *aligned;
    int64_t offset, size[3], stride[3];
} Memref;

typedef void (*Kernel)(Memref *result, Memref *input);

static void read_values(const char *name, float *values) {
    FILE *file = fopen(name, "rb");
    if (!file || fread(values, sizeof(float), count, file) != count) {
        fprintf(stderr, "Could not read %s\n", name);
        exit(1);
    }
    fclose(file);
}

static double seconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void bench_driver(Kernel kernel) {
    float x[count], expected[count];
    read_values("X.f32", x);
    read_values("Y.f32", expected);
    Memref input = {x, x, 0, {1, tokens, features}, {count, features, 1}};
    Memref result;
    kernel(&result, &input);  // Warm up and check the actual compiled kernel.

    if (result.size[0] != 1 || result.size[1] != tokens || result.size[2] != features) {
        fprintf(stderr, "Unexpected FFN result shape\n");
        exit(1);
    }
    for (int t = 0; t < tokens; ++t) {
        for (int f = 0; f < features; ++f) {
            double actual = result.aligned[result.offset + t*result.stride[1] + f*result.stride[2]];
            double ref = expected[t*features + f];
            if (!isfinite(actual) || fabs(actual-ref) > 1e-5 + 1e-4*fabs(ref)) {
                fprintf(stderr, "Mismatch at [0,%d,%d]: got %.9g, expected %.9g\n", t, f, actual, ref);
                exit(1);
            }
        }
    }
    free(result.allocated);

    fputs("  Warm-up output matches Y. Timing 100 calls...\n", stderr);
    fflush(stderr);
    double start = seconds();
    for (int i = 0; i < repeats; ++i) {
        kernel(&result, &input);
        free(result.allocated);
    }
    double elapsed = seconds() - start;
    printf("%.9f\n", elapsed * 1000.0 / repeats);
}
"""

# Pass the kernel's C entry point to the native driver from inside mlir-runner.
ENTRY = """
  llvm.func @bench_driver(!llvm.ptr)
  llvm.func @bench_entry() {
    %kernel = llvm.mlir.addressof @_mlir_ciface_ffn_kernel : !llvm.ptr
    llvm.call @bench_driver(%kernel) : (!llvm.ptr) -> ()
    llvm.return
  }
"""

PASSES = [
    "--empty-tensor-to-alloc-tensor",
    "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
    "--convert-linalg-to-loops",
    "--buffer-deallocation-pipeline",
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--convert-math-to-libm",
    "--convert-math-to-llvm",
    "--convert-arith-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--reconcile-unrealized-casts",
]


def lower(source, directory):
    # Both exported modules contain one public @main taking and returning a tensor.
    print(f"Lowering {source.name}...", flush=True)
    text = source.read_text()
    signature = r"\bfunc\.func\s+@main\s*(\([^)]*\)\s*->\s*tensor<1x64x128xf32>)\s*\{"
    text, matches = re.subn(signature, r"func.func @ffn_kernel\1 attributes {llvm.emit_c_interface} {", text)
    if matches != 1: