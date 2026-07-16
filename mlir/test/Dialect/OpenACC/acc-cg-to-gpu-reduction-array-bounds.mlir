// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// CHECK-LABEL: func.func @dynamic_private_local
// CHECK: memref.subview
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
func.func @dynamic_private_local(%n: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize(%n) [#acc<par_dims[block_x, thread_x]>] : (index) -> !acc.private_type<memref<?xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg0 = %private, %extent = %n) : (!acc.private_type<memref<?xi32>>, index) {
    %local = acc.private_local %arg0 : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    %bounds = acc.bounds extent(%extent : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add> : memref<?xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @strided_extent
// CHECK: %[[STEP:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
// CHECK: %[[SPAN:.*]] = arith.muli %{{.*}}, %[[STEP]] : index
// CHECK: %[[UB:.*]] = arith.addi %{{.*}}, %[[SPAN]] : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[UB]] step %[[STEP]]
func.func @strided_extent(%lb: index, %extent: index, %byte_stride: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%argLb = %lb, %argExtent = %extent, %argStride = %byte_stride) : (index, index, index) {
    %array = memref.alloca() : memref<64xi32>
    %bounds = acc.bounds lowerbound(%argLb : index) extent(%argExtent : index) stride(%argStride : index) {strideInBytes = true}
    acc.reduction_accumulate_array %array bounds(%bounds) <add> : memref<64xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @compact_normalized
// CHECK: %[[SPAN:.*]] = arith.subi %{{.*}}, %{{.*}} : index
// CHECK: %[[COUNT0:.*]] = arith.divsi %[[SPAN]], %{{.*}} : index
// CHECK: %[[COUNT:.*]] = arith.addi %[[COUNT0]], %{{.*}} : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[COUNT]] step %{{.*}}
// CHECK: gpu.all_reduce add
func.func @compact_normalized(%lb: index, %ub: index, %stride: index, %size: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%argLb = %lb, %argUb = %ub, %argStride = %stride, %argSize = %size) : (index, index, index, index) {
    %array = memref.alloca(%argSize) : memref<?xi32>
    %bounds = acc.bounds lowerbound(%argLb : index) upperbound(%argUb : index) stride(%argStride : index)
    acc.reduction_accumulate_array %array bounds(%bounds) <add> : memref<?xi32> {accumulator_is_compact, accumulator_is_thread_private, par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @inclusive_upperbound
// CHECK: gpu.launch
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[UB:.*]] = arith.addi %{{.*}}, %[[ONE]] : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[UB]] step %{{.*}}
func.func @inclusive_upperbound(%lb: index, %ub: index, %stride: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%argLb = %lb, %argUb = %ub, %argStride = %stride) : (index, index, index) {
    %array = memref.alloca() : memref<64xi32>
    %bounds = acc.bounds lowerbound(%argLb : index) upperbound(%argUb : index) stride(%argStride : index)
    acc.reduction_accumulate_array %array bounds(%bounds) <add> : memref<64xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}
