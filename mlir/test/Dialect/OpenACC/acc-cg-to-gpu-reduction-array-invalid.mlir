// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-prepare-cg-to-gpu,acc-cg-to-gpu))" \
// RUN:   -verify-diagnostics

func.func @dynamic_byte_stride(
    %buffer: memref<?xi32>, %extent: index, %byte_stride: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%ext = %extent, %step = %byte_stride) : (index, index) {
    %local = memref.alloca(%ext) : memref<?xi32>
    %bounds = acc.bounds extent(%ext : index)
        stride(%step : index) {strideInBytes = true}
    // expected-error@+1 {{reduction: dynamic byte stride requires alignment validation}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<?xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @misaligned_byte_stride() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<3xi32>
    %bounds = acc.bounds extent(%c3 : index)
        stride(%c2 : index) {strideInBytes = true}
    // expected-error@+1 {{reduction: byte stride is not element-aligned}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<3xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @zero_stride() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c0 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<3xi32>
    %bounds = acc.bounds extent(%c3 : index) stride(%c0 : i32)
    // expected-error@+1 {{reduction: zero array stride}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<3xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @minimum_signed_stride() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c7 = arith.constant 7 : index
    %cmin = arith.constant -9223372036854775808 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds extent(%c7 : index) stride(%cmin : index)
    // expected-error@+1 {{reduction: minimum signed array stride}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @thread_y_only_array_reduction(%extent: index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %ty = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  acc.compute_region launch(%kbx = %bx, %kty = %ty)
      ins(%ext = %extent) : (index) {
    %local = memref.alloca(%ext) : memref<?xi32>
    %bounds = acc.bounds extent(%ext : index)
    // expected-error@+1 {{reduction: array reduction does not span all thread dimensions}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<?xi32> {par_dims = #acc<par_dims[block_x, thread_y]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @pointer_like_array_reduction(%pointer: !llvm.ptr) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%ptr = %pointer) : (!llvm.ptr) {
    %extent = arith.constant 1 : index
    %bounds = acc.bounds extent(%extent : index)
    // expected-error@+1 {{reduction: non-MemRefTy accumulate array}}
    acc.reduction_accumulate_array %ptr bounds(%bounds) <add>
        : !llvm.ptr {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

