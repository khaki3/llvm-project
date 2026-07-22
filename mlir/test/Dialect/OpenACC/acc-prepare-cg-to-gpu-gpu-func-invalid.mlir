// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-prepare-cg-to-gpu)))" \
// RUN:   -verify-diagnostics

gpu.module @device {
  gpu.func @unsupported_argument(%shared: memref<2xi32>) {
    %c2 = arith.constant 2 : index
    %bounds = acc.bounds extent(%c2 : index)
    // expected-error@+1 {{mixed or unsupported array storage ownership}}
    acc.reduction_accumulate_array %shared bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    gpu.return
  }
}
