// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// CHECK-LABEL: func.func @explicit_per_thread
// CHECK: %[[LB:.*]] = arith.constant 5 : index
// CHECK: %[[START:.*]] = arith.constant 1 : index
// CHECK: %[[NORMALIZED:.*]] = arith.subi %[[LB]], %[[START]] : index
// CHECK: %[[INDEX:.*]] = arith.addi %[[NORMALIZED]], %{{.*}} : index
// CHECK: gpu.all_reduce add
func.func @explicit_per_thread() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c5 = arith.constant 5 : index
    %start = arith.constant 1 : index
    %extent = arith.constant 2 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds lowerbound(%c5 : index) extent(%extent : index)
        startIdx(%start : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32>
        {acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>,
         acc.array_reduction_operator = #acc.reduction_operator<add>,
         par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @explicit_shared_noop
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
func.func @explicit_shared_noop(%shared: memref<2xi32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %shared) : (memref<2xi32>) {
    %extent = arith.constant 2 : index
    %bounds = acc.bounds extent(%extent : index)
    acc.reduction_accumulate_array %arg bounds(%bounds) <add>
        : memref<2xi32>
        {acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>,
         acc.array_reduction_operator = #acc.reduction_operator<add>,
         par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}
