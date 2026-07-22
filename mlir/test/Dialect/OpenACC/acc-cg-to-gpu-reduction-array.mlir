// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-prepare-cg-to-gpu,acc-cg-to-gpu))" | FileCheck %s

// A per-thread array accumulator (memref.alloca) for a block+thread reduction is
// reduced element-by-element across the parallel dimensions with gpu.all_reduce -
// the array analog of the scalar acc.reduction_accumulate.

// CHECK-LABEL: func.func @array_reduction
// CHECK: gpu.launch
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
// CHECK-NOT: acc.reduction_accumulate_array
// CHECK-NOT: acc.bounds
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[OFFSET:.*]] = arith.muli %[[IV]], %{{.*}} : index
// CHECK:   %[[INDEX:.*]] = arith.addi %{{.*}}, %[[OFFSET]] : index
// CHECK:   %[[ELT:.*]] = memref.load %[[ALLOCA]][%[[INDEX]]] : memref<2xi32>
// CHECK:   %[[RED:.*]] = gpu.all_reduce add %[[ELT]]
// Per-thread alloca: the all_reduce result is stored unpredicated.
// CHECK:   memref.store %[[RED]], %[[ALLOCA]][%[[INDEX]]] : memref<2xi32>
// CHECK: }

func.func @array_reduction(%arg0: memref<2xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<2xi32>) -> memref<2xi32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "r"}
  acc.kernel_environment dataOperands(%0 : memref<2xi32>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw {par_dim = #acc.par_dim<block_x>}
    %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<2xi32>) {
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %2 = acc.reduction_init %arg2 <add> : memref<2xi32> {
        %alloca = memref.alloca() : memref<2xi32>
        scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
          memref.store %c0_i32, %alloca[%i] : memref<2xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloca : memref<2xi32>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %3 = memref.load %2[%c0] : memref<2xi32>
          %4 = arith.addi %3, %c1_i32 : i32
          memref.store %4, %2[%c0] : memref<2xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c2 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
      acc.reduction_combine_region %2 into %arg2 : memref<2xi32> {
        scf.for %i = %c0 to %c2 step %c1 {
          %3 = memref.load %2[%i] : memref<2xi32>
          %4 = memref.load %arg2[%i] : memref<2xi32>
          %5 = arith.addi %3, %4 : i32
          memref.store %5, %arg2[%i] : memref<2xi32>
        }
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%0 : memref<2xi32>) to varPtr(%arg0 : memref<2xi32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "r"}
  return
}

// CHECK-LABEL: func.func @array_reduction_small_shared
// CHECK: memref.alloc() : memref<2xi32>
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
func.func @array_reduction_small_shared() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c2 = arith.constant 2 : index
    %shared = memref.alloc() : memref<2xi32>
    %bounds = acc.bounds extent(%c2 : index)
    acc.reduction_accumulate_array %shared bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    memref.dealloc %shared : memref<2xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_dynamic_shared
// CHECK: memref.alloc(%{{.*}}) : memref<?xi32>
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
func.func @array_reduction_dynamic_shared(%n: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ext = %n) : (index) {
    %shared = memref.alloc(%ext) : memref<?xi32>
    %bounds = acc.bounds extent(%ext : index)
    acc.reduction_accumulate_array %shared bounds(%bounds) <add>
        : memref<?xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    memref.dealloc %shared : memref<?xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_strided_extent
// CHECK: gpu.launch
// CHECK: %[[LB:.*]] = arith.constant 1 : index
// CHECK: %[[STEP:.*]] = arith.constant 2 : index
// CHECK: %[[EXTENT:.*]] = arith.constant 3 : index
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[ITER:.*]] = %[[ZERO]] to %[[EXTENT]] step %[[ONE]]
// CHECK:   %[[OFFSET:.*]] = arith.muli %[[ITER]], %[[STEP]] : index
// CHECK:   %[[INDEX:.*]] = arith.addi %[[LB]], %[[OFFSET]] : index
// CHECK:   memref.load %{{.*}}[%[[INDEX]]]
func.func @array_reduction_strided_extent() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c1_b = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds lowerbound(%c1_b : index) extent(%c3 : index)
        stride(%c2 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// Negative strides use a positive normalized loop and calculate the descending
// memref index in the body.
//
// CHECK-LABEL: func.func @array_reduction_negative_stride
// CHECK: %[[LB:.*]] = arith.constant 5 : index
// CHECK: %[[STEP:.*]] = arith.constant -1 : index
// CHECK: %[[EXTENT:.*]] = arith.constant 3 : index
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[ITER:.*]] = %[[ZERO]] to %[[EXTENT]] step %[[ONE]]
// CHECK:   %[[OFFSET:.*]] = arith.muli %[[ITER]], %[[STEP]] : index
// CHECK:   %[[INDEX:.*]] = arith.addi %[[LB]], %[[OFFSET]] : index
// CHECK:   memref.load %{{.*}}[%[[INDEX]]]
func.func @array_reduction_negative_stride() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c5 = arith.constant 5 : index
    %cneg1 = arith.constant -1 : index
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds lowerbound(%c5 : index) extent(%c3 : index)
        stride(%cneg1 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_byte_stride
// CHECK: arith.constant 4 : index
// CHECK: memref.alloca() : memref<3xi32>
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 1 : index
// CHECK: %[[ELEMENT_STEP:.*]] = arith.constant 1 : index
// CHECK-NEXT: scf.for %[[ITER:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:   %[[OFFSET:.*]] = arith.muli %[[ITER]], %[[ELEMENT_STEP]] : index
// CHECK:   %[[INDEX:.*]] = arith.addi %{{.*}}, %[[OFFSET]] : index
// CHECK:   memref.load %{{.*}}[%[[INDEX]]]
func.func @array_reduction_byte_stride() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %local = memref.alloca() : memref<3xi32>
    %bounds = acc.bounds lowerbound(%c0 : index) extent(%c3 : index)
        stride(%c4 : index) {strideInBytes = true}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<3xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_upperbound_negative_stride
// CHECK: %[[LB:.*]] = arith.constant 5 : index
// CHECK: %[[STEP:.*]] = arith.constant -1 : index
// CHECK: %[[UB:.*]] = arith.constant 3 : index
// CHECK: %[[DISTANCE:.*]] = arith.subi %[[LB]], %[[UB]] : index
// CHECK: %[[DIVISOR:.*]] = arith.constant 1 : index
// CHECK: %[[SPAN:.*]] = arith.divsi %[[DISTANCE]], %[[DIVISOR]] : index
// CHECK: %[[NONNEGATIVE:.*]] = arith.cmpi sge, %[[DISTANCE]], %{{.*}} : index
// CHECK: %[[TRIP_COUNT:.*]] = arith.addi %[[SPAN]], %{{.*}} : index
// CHECK: %[[COUNT:.*]] = arith.select %[[NONNEGATIVE]], %[[TRIP_COUNT]], %{{.*}} : index
// CHECK: scf.for %[[ITER:.*]] = %{{.*}} to %[[COUNT]] step %{{.*}}
// CHECK:   %[[OFFSET:.*]] = arith.muli %[[ITER]], %[[STEP]] : index
// CHECK:   %[[INDEX:.*]] = arith.addi %[[LB]], %[[OFFSET]] : index
func.func @array_reduction_upperbound_negative_stride() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c5 = arith.constant 5 : index
    %cneg1 = arith.constant -1 : index
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds lowerbound(%c5 : index) upperbound(%c3 : index)
        stride(%cneg1 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// An upperbound range whose bounds run opposite to its stride has zero
// iterations.
//
// CHECK-LABEL: func.func @array_reduction_empty_upperbound_range
// CHECK: %[[LB:.*]] = arith.constant 5 : index
// CHECK: %[[UB:.*]] = arith.constant 3 : index
// CHECK: %[[DISTANCE:.*]] = arith.subi %[[UB]], %[[LB]] : index
// CHECK: %[[NONNEGATIVE:.*]] = arith.cmpi sge, %[[DISTANCE]], %[[ZERO:.*]] : index
// CHECK: %[[COUNT:.*]] = arith.select %[[NONNEGATIVE]], %{{.*}}, %[[ZERO]] : index
// CHECK: scf.for %{{.*}} = %[[ZERO]] to %[[COUNT]] step %{{.*}}
func.func @array_reduction_empty_upperbound_range() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds lowerbound(%c5 : index) upperbound(%c3 : index)
        stride(%c4 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// A dynamically-shaped private uses indexed global storage, whose thread
// slices are combined directly without a redundant gpu.all_reduce.
//
// CHECK-LABEL: func.func @array_reduction_dynamic_private
// CHECK: arith.select
// CHECK-NOT: gpu.all_reduce
func.func @array_reduction_dynamic_private(%n: index, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize(%n) [#acc<par_dims[block_x, thread_x]>]
      : (index) -> !acc.private_type<memref<?xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%private_arg = %private, %ext = %n, %condition_arg = %condition)
      : (!acc.private_type<memref<?xi32>>, index, i1) {
    %local = acc.private_local %private_arg
        : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    %selected = arith.select %condition_arg, %local, %local : memref<?xi32>
    %bounds = acc.bounds extent(%ext : index)
    acc.reduction_accumulate_array %selected bounds(%bounds) <add>
        : memref<?xi32>
        {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// An external dynamic memref has no proven per-thread ownership, even when the
// reduction itself includes ThreadX.
//
// CHECK-LABEL: func.func @array_reduction_dynamic_external
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
func.func @array_reduction_dynamic_external(
    %buffer: memref<?xi32>, %n: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg0 = %buffer, %ext = %n) : (memref<?xi32>, index) {
    %bounds = acc.bounds extent(%ext : index)
    acc.reduction_accumulate_array %arg0 bounds(%bounds) <add>
        : memref<?xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// A constant select preserves the ownership of its selected per-thread target.
//
// CHECK-LABEL: func.func @array_reduction_selected_private
// CHECK: arith.select
// CHECK: gpu.all_reduce add
func.func @array_reduction_selected_private(%shared: memref<2xi32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%shared_arg = %shared) : (memref<2xi32>) {
    %local = memref.alloca() : memref<2xi32>
    %true = arith.constant true
    %selected = arith.select %true, %local, %shared_arg : memref<2xi32>
    %c2 = arith.constant 2 : index
    %bounds = acc.bounds extent(%c2 : index)
    acc.reduction_accumulate_array %selected bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_wide_integer_stride
// CHECK: gpu.all_reduce add
func.func @array_reduction_wide_integer_stride() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c2 = arith.constant 2 : index
    %wide_step = arith.constant 1 : i128
    %local = memref.alloca() : memref<2xi32>
    %bounds = acc.bounds extent(%c2 : index) stride(%wide_step : i128)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_dynamic_byte_stride_i8
// CHECK: gpu.all_reduce add
func.func @array_reduction_dynamic_byte_stride_i8(%stride: index) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%step = %stride) : (index) {
    %c8 = arith.constant 8 : index
    %local = memref.alloca() : memref<8xi8>
    %bounds = acc.bounds extent(%c8 : index)
        stride(%step : index) {strideInBytes = true}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi8> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_scf_if_private
// CHECK-NOT: gpu.all_reduce
func.func @array_reduction_scf_if_private(%n: index, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize(%n) [#acc<par_dims[block_x, thread_x]>]
      : (index) -> !acc.private_type<memref<?xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%private_arg = %private, %ext = %n, %condition_arg = %condition)
      : (!acc.private_type<memref<?xi32>>, index, i1) {
    %local = acc.private_local %private_arg
        : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    %unused, %selected = scf.if %condition_arg
        -> (memref<?xi32>, memref<?xi32>) {
      scf.yield %local, %local : memref<?xi32>, memref<?xi32>
    } else {
      scf.yield %local, %local : memref<?xi32>, memref<?xi32>
    }
    %bounds = acc.bounds extent(%ext : index)
    acc.reduction_accumulate_array %selected bounds(%bounds) <add>
        : memref<?xi32>
        {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_worker_row
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: gpu.all_reduce add
// CHECK: scf.if
// CHECK: memref.store
func.func @array_reduction_worker_row(%n: index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %ty = acc.par_width %c2 {par_dim = #acc.par_dim<thread_y>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize(%n) [#acc<par_dims[thread_y]>]
      : (index) -> !acc.private_type<memref<?xi32>>
  acc.compute_region launch(%kbx = %bx, %kty = %ty, %ktx = %tx)
      ins(%private_arg = %private, %extent = %n)
      : (!acc.private_type<memref<?xi32>>, index) {
    %local = acc.private_local %private_arg
        {acc.par_dims = #acc<par_dims[thread_y]>}
        : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    %bounds = acc.bounds extent(%extent : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<?xi32>
        {par_dims = #acc<par_dims[block_x, thread_y, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}
