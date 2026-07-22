// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-prepare-cg-to-gpu))" \
// RUN:   -verify-diagnostics

func.func @mixed_storage(%shared: memref<2xi32>, %condition: i1) {
  %local = memref.alloca() : memref<2xi32>
  %selected = arith.select %condition, %local, %shared : memref<2xi32>
  %c2 = arith.constant 2 : index
  %bounds = acc.bounds extent(%c2 : index)
  // expected-error@+1 {{mixed or unsupported array storage ownership}}
  acc.reduction_accumulate_array %selected bounds(%bounds) <add>
      : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

func.func @stale_shared_mode_on_alloca() {
  %c2 = arith.constant 2 : index
  %local = memref.alloca() : memref<2xi32>
  %bounds = acc.bounds extent(%c2 : index)
  // expected-error@+1 {{invalid precomputed array reduction policy}}
  acc.reduction_accumulate_array %local bounds(%bounds) <add>
      : memref<2xi32>
      {acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>,
       acc.array_reduction_operator = #acc.reduction_operator<add>,
       par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

func.func @stale_thread_mode_on_shared(%shared: memref<2xi32>) {
  %c2 = arith.constant 2 : index
  %bounds = acc.bounds extent(%c2 : index)
  // expected-error@+1 {{invalid precomputed array reduction policy}}
  acc.reduction_accumulate_array %shared bounds(%bounds) <add>
      : memref<2xi32>
      {acc.array_reduction_mode = #acc.array_reduction_mode<per_thread>,
       acc.array_reduction_operator = #acc.reduction_operator<add>,
       par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

func.func @worker_row_without_thread_y(%shared: memref<2xi32>) {
  %c2 = arith.constant 2 : index
  %opaque = scf.execute_region -> memref<2xi32> {
    scf.yield %shared : memref<2xi32>
  }
  %bounds = acc.bounds extent(%c2 : index)
  // expected-error@+1 {{invalid precomputed array reduction policy}}
  acc.reduction_accumulate_array %opaque bounds(%bounds) <add>
      : memref<2xi32>
      {acc.array_reduction_mode = #acc.array_reduction_mode<worker_row>,
       acc.array_reduction_operator = #acc.reduction_operator<add>,
       par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

func.func @inferred_per_thread_without_thread_x() {
  %c2 = arith.constant 2 : index
  %local = memref.alloca() : memref<2xi32>
  %bounds = acc.bounds extent(%c2 : index)
  // expected-error@+1 {{array reduction policy is incompatible with parallel dimensions}}
  acc.reduction_accumulate_array %local bounds(%bounds) <add>
      : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_y]>}
  return
}

func.func @inferred_worker_row_without_thread_x() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %ty = acc.par_width %c2 {par_dim = #acc.par_dim<thread_y>}
  %private = acc.privatize [#acc<par_dims[thread_y]>]
      : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region launch(%kbx = %bx, %kty = %ty)
      ins(%arg = %private) : (!acc.private_type<memref<2xi32>>) {
    %c2_inner = arith.constant 2 : index
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[thread_y]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %bounds = acc.bounds extent(%c2_inner : index)
    // expected-error@+1 {{array reduction policy is incompatible with parallel dimensions}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_y]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @worker_row_with_thread_z() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %ty = acc.par_width %c2 {par_dim = #acc.par_dim<thread_y>}
  %tz = acc.par_width %c2 {par_dim = #acc.par_dim<thread_z>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[thread_y]>]
      : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region launch(%kbx = %bx, %kty = %ty, %ktz = %tz, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<2xi32>>) {
    %c2_inner = arith.constant 2 : index
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[thread_y]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %bounds = acc.bounds extent(%c2_inner : index)
    // expected-error@+1 {{array reduction policy is incompatible with parallel dimensions}}
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<2xi32>
        {par_dims = #acc<par_dims[block_x, thread_y, thread_z, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @conflicting_operators() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<2xi32>>) {
    %c2 = arith.constant 2 : index
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[thread_x]>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    %bounds = acc.bounds extent(%c2 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    // expected-error@+1 {{incompatible array reduction operators}}
    acc.reduction_accumulate_array %local bounds(%bounds) <mul>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @rank_two_array() {
  %c2 = arith.constant 2 : index
  %local = memref.alloca() : memref<2x2xi32>
  %bounds = acc.bounds extent(%c2 : index)
  // expected-error@+1 {{array reduction requires a rank-1 memref}}
  acc.reduction_accumulate_array %local bounds(%bounds) <add>
      : memref<2x2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}

func.func @rank_changing_private_view() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[thread_x]>]
      : () -> !acc.private_type<memref<2x2xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<2x2xi32>>) {
    %c4 = arith.constant 4 : index
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[thread_x]>}
        : (!acc.private_type<memref<2x2xi32>>) -> memref<2x2xi32>
    %flat = memref.collapse_shape %local [[0, 1]]
        : memref<2x2xi32> into memref<4xi32>
    %bounds = acc.bounds extent(%c4 : index)
    // expected-error@+1 {{array reduction private storage must be rank-1}}
    acc.reduction_accumulate_array %flat bounds(%bounds) <add>
        : memref<4xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @thread_only_shared_array_reduction(%shared: memref<2xi32>) {
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%ktx = %tx)
      ins(%arg = %shared, %extent = %c2) : (memref<2xi32>, index) {
    %bounds = acc.bounds extent(%extent : index)
    // expected-error@+1 {{reduction: thread-only array reduction accumulate}}
    acc.reduction_accumulate_array %arg bounds(%bounds) <add>
        : memref<2xi32>
        {acc.array_reduction_mode = #acc.array_reduction_mode<block_shared_noop>,
         acc.array_reduction_operator = #acc.reduction_operator<add>,
         par_dims = #acc<par_dims[thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

func.func @loop_carried_mixed(%shared: memref<2xi32>, %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %local = memref.alloca() : memref<2xi32>
  %bounds = acc.bounds extent(%c2 : index)
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg = %local)
      -> (memref<2xi32>) {
    // expected-error@+1 {{mixed or unsupported array storage ownership}}
    acc.reduction_accumulate_array %arg bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    %selected = arith.select %condition, %arg, %shared : memref<2xi32>
    scf.yield %selected : memref<2xi32>
  }
  return
}
