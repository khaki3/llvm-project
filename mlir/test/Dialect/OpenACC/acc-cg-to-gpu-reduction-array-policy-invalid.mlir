// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" \
// RUN:   -split-input-file -verify-diagnostics

func.func @missing_private_operator() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  // expected-error@+1 {{failed to legalize operation 'acc.compute_region'}}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<2xi32>>) {
    // expected-error@+1 {{prepared array reduction storage requires a reduction operator}}
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[thread_x]>,
         acc.private_storage_kind = #acc.private_storage_kind<per_thread>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

func.func @stale_private_operator() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  // expected-error@+1 {{failed to legalize operation 'acc.compute_region'}}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<2xi32>>) {
    %c2 = arith.constant 2 : index
    %local = acc.private_local %arg
        {acc.array_reduction_operator = #acc.reduction_operator<mul>,
         acc.par_dims = #acc<par_dims[thread_x]>,
         acc.private_storage_kind = #acc.private_storage_kind<per_thread>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %alias = memref.cast %local : memref<2xi32> to memref<?xi32>
    %bounds = acc.bounds extent(%c2 : index)
    // expected-error@+1 {{prepared array reduction operator does not match accumulate}}
    acc.reduction_accumulate_array %alias bounds(%bounds) <add>
        : memref<?xi32>
        {acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>,
         acc.array_reduction_operator = #acc.reduction_operator<mul>,
         par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

func.func @zero_sized_shared_storage() {
  %c1 = arith.constant 1 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %private = acc.privatize : () -> !acc.private_type<memref<0xi32>>
  // expected-error@+1 {{failed to legalize operation 'acc.compute_region'}}
  acc.compute_region launch(%kbx = %bx)
      ins(%arg = %private) : (!acc.private_type<memref<0xi32>>) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{prepared shared storage cannot be zero-sized}}
    %local = acc.private_local %arg
        {acc.array_reduction_operator = #acc.reduction_operator<add>,
         acc.private_storage_kind = #acc.private_storage_kind<block_shared>}
        : (!acc.private_type<memref<0xi32>>) -> memref<0xi32>
    %bounds = acc.bounds extent(%c0 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<0xi32>
        {acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>,
         acc.array_reduction_operator = #acc.reduction_operator<add>,
         par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}
