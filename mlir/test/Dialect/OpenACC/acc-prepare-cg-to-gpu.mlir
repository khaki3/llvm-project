// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-prepare-cg-to-gpu,acc-prepare-cg-to-gpu))" | FileCheck %s

// CHECK-LABEL: func.func @direct_storage
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>
func.func @direct_storage(%shared: memref<4xi32>) {
  %c4 = arith.constant 4 : index
  %local = memref.alloca() : memref<4xi32>
  %bounds = acc.bounds extent(%c4 : index)
  acc.reduction_accumulate_array %local bounds(%bounds) <add>
      : memref<4xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  acc.reduction_accumulate_array %shared bounds(%bounds) <add>
      : memref<4xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

// CHECK-LABEL: func.func @large_indexed_private
// CHECK: acc.private_local
// CHECK-SAME: acc.private_storage_kind = #acc.private_storage_kind<indexed_global>
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>
func.func @large_indexed_private() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[block_x, thread_x]>]
      : () -> !acc.private_type<memref<8192xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<8192xi32>>) {
    %c8192 = arith.constant 8192 : index
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
        : (!acc.private_type<memref<8192xi32>>) -> memref<8192xi32>
    %bounds = acc.bounds extent(%c8192 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8192xi32>
        {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @block_only_private
// CHECK: acc.private_local
// CHECK-SAME: acc.array_reduction_operator = #acc.reduction_operator<add>
// CHECK-SAME: acc.private_storage_kind = #acc.private_storage_kind<indexed_global>
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>
func.func @block_only_private() {
  %c1 = arith.constant 1 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %private = acc.privatize [#acc<par_dims[block_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region launch(%kbx = %bx)
      ins(%arg = %private) : (!acc.private_type<memref<2xi32>>) {
    %c2 = arith.constant 2 : index
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[block_x]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %bounds = acc.bounds extent(%c2 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @matching_worker_if
// CHECK-COUNT-2: acc.private_storage_kind = #acc.private_storage_kind<indexed_global>
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<worker_row>
func.func @matching_worker_if(%n: index, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %ty = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private0 = acc.privatize(%n) [#acc<par_dims[thread_y]>]
      : (index) -> !acc.private_type<memref<?xi32>>
  %private1 = acc.privatize(%n) [#acc<par_dims[thread_y]>]
      : (index) -> !acc.private_type<memref<?xi32>>
  acc.compute_region launch(%kbx = %bx, %kty = %ty, %ktx = %tx)
      ins(%arg0 = %private0, %arg1 = %private1, %extent = %n,
          %cond = %condition)
      : (!acc.private_type<memref<?xi32>>,
         !acc.private_type<memref<?xi32>>, index, i1) {
    %local0 = acc.private_local %arg0
        {acc.par_dims = #acc<par_dims[thread_y]>}
        : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    %local1 = acc.private_local %arg1
        {acc.par_dims = #acc<par_dims[thread_y]>}
        : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    %unused, %selected = scf.if %cond
        -> (memref<?xi32>, memref<?xi32>) {
      scf.yield %local0, %local0 : memref<?xi32>, memref<?xi32>
    } else {
      scf.yield %local1, %local1 : memref<?xi32>, memref<?xi32>
    }
    %bounds = acc.bounds extent(%extent : index)
    acc.reduction_accumulate_array %selected bounds(%bounds) <add>
        : memref<?xi32>
        {par_dims = #acc<par_dims[block_x, thread_y, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @constant_select
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>
func.func @constant_select(%shared: memref<2xi32>) {
  %true = arith.constant true
  %local = memref.alloca() : memref<2xi32>
  %selected = arith.select %true, %local, %shared : memref<2xi32>
  %c2 = arith.constant 2 : index
  %bounds = acc.bounds extent(%c2 : index)
  acc.reduction_accumulate_array %selected bounds(%bounds) <add>
      : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

// CHECK-LABEL: func.func @loop_carried_private
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>
func.func @loop_carried_private() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %local = memref.alloca() : memref<2xi32>
  %bounds = acc.bounds extent(%c2 : index)
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg = %local)
      -> (memref<2xi32>) {
    acc.reduction_accumulate_array %arg bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    scf.yield %arg : memref<2xi32>
  }
  return
}

// CHECK-LABEL: func.func @loop_carried_private_cast
// CHECK-COUNT-2: acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>
func.func @loop_carried_private_cast() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %local = memref.alloca() : memref<2xi32>
  %bounds = acc.bounds extent(%c2 : index)
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg = %local)
      -> (memref<2xi32>) {
    %cast = memref.cast %arg : memref<2xi32> to memref<2xi32>
    acc.reduction_accumulate_array %cast bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    scf.yield %cast : memref<2xi32>
  }
  acc.reduction_accumulate_array %result bounds(%bounds) <add>
      : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

// CHECK-LABEL: func.func @loop_carried_private_rotation
// CHECK-COUNT-3: acc.private_local{{.*}}acc.array_reduction_operator = #acc.reduction_operator<add>{{.*}}acc.private_storage_kind = #acc.private_storage_kind<per_thread>
// CHECK: acc.reduction_accumulate_array
// CHECK-SAME: acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>
// CHECK-SAME: acc.array_reduction_operator = #acc.reduction_operator<add>
func.func @loop_carried_private_rotation() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private0 = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  %private1 = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  %private2 = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg0 = %private0, %arg1 = %private1, %arg2 = %private2)
      : (!acc.private_type<memref<2xi32>>,
         !acc.private_type<memref<2xi32>>,
         !acc.private_type<memref<2xi32>>) {
    %c0 = arith.constant 0 : index
    %c1loop = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %local0 = acc.private_local %arg0
        {acc.par_dims = #acc<par_dims[thread_x]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %local1 = acc.private_local %arg1
        {acc.par_dims = #acc<par_dims[thread_x]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %local2 = acc.private_local %arg2
        {acc.par_dims = #acc<par_dims[thread_x]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %result0, %result1, %result2 = scf.for %i = %c0 to %c1loop step %c1loop
        iter_args(%iter0 = %local0, %iter1 = %local1, %iter2 = %local2)
        -> (memref<2xi32>, memref<2xi32>, memref<2xi32>) {
      %next0 = memref.cast %iter1 : memref<2xi32> to memref<2xi32>
      %next1 = memref.cast %iter2 : memref<2xi32> to memref<2xi32>
      %next2 = memref.cast %iter0 : memref<2xi32> to memref<2xi32>
      scf.yield %next0, %next1, %next2
          : memref<2xi32>, memref<2xi32>, memref<2xi32>
    }
    %bounds = acc.bounds extent(%c2 : index)
    acc.reduction_accumulate_array %result2 bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}
