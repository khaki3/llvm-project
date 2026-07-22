// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A worker-private reduction has one shared slot per ThreadY row. The combine
// must keep ThreadY active and predicate only ThreadX.

// CHECK-LABEL: func.func @worker_reduction_combine
// CHECK: gpu.launch {{.*}} threads([[TID_X:%[^,]+]], [[TID_Y:%[^,]+]],
// CHECK-NOT: arith.cmpi eq, [[TID_Y]]
// CHECK: arith.select
// CHECK: %[[IS_X_ZERO:.*]] = arith.cmpi eq, [[TID_X]],
// CHECK-NOT: arith.andi
// CHECK: scf.if %[[IS_X_ZERO]]
// CHECK: acc.atomic.update

func.func @worker_reduction_combine(%result: memref<i32>, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %result_arg = %result,
            %condition_arg = %condition)
        : (!acc.private_type<memref<i32>>, memref<i32>, i1) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %true = arith.constant true
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c0_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %cast = memref.cast %local
            : memref<i32> to memref<i32, strided<[], offset: ?>>
        %selected = arith.select %condition_arg, %cast, %cast
            : memref<i32, strided<[], offset: ?>>
        %store_target = arith.select %true, %local, %result_arg : memref<i32>
        %result_cast = memref.cast %result_arg
            : memref<i32> to memref<i32, strided<[], offset: ?>>
        acc.predicate_region {
          %unused = memref.load %result_arg[] : memref<i32>
          memref.store %c0_i32, %store_target[] : memref<i32>
          acc.reduction_combine %selected into %result_cast <add>
              : memref<i32, strided<[], offset: ?>>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// Nested predicate regions choose their ThreadY predicates independently. The
// outer region must keep ThreadY active so it does not exclude worker rows
// before the nested worker-private combine is reached.
// CHECK-LABEL: func.func @nested_worker_reduction_combines
// CHECK: gpu.launch {{.*}} threads([[NESTED_TX:%[^,]+]], [[NESTED_TY:%[^,]+]],
// CHECK-NOT: arith.cmpi eq, [[NESTED_TY]]
// CHECK: %[[NESTED_TX_ZERO:.*]] = arith.cmpi eq, [[NESTED_TX]],
// CHECK-NOT: arith.andi
// CHECK: scf.if %[[NESTED_TX_ZERO]]
func.func @nested_worker_reduction_combines(
    %other: memref<i32>, %result: memref<i32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %other_arg = %other,
            %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c0_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        acc.predicate_region {
          acc.predicate_region {
            acc.reduction_combine %local into %result_arg <add> : memref<i32>
                {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
          }
          acc.predicate_region {
            acc.reduction_combine %other_arg into %result_arg <add> : memref<i32>
                {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
          }
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// CHECK-LABEL: func.func @worker_combine_in_scf_if
// CHECK: gpu.launch {{.*}} threads([[IF_TX:%[^,]+]], [[IF_TY:%[^,]+]],
// CHECK-NOT: arith.cmpi eq, [[IF_TY]]
// CHECK: %[[IF_TX_ZERO:.*]] = arith.cmpi eq, [[IF_TX]],
// CHECK-NOT: arith.andi
// CHECK: scf.if %[[IF_TX_ZERO]]
func.func @worker_combine_in_scf_if(%result: memref<i32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %true = arith.constant true
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        acc.predicate_region {
          scf.if %true {
            acc.reduction_combine %local into %result_arg <add> : memref<i32>
                {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
          }
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// CHECK-LABEL: func.func @worker_combine_scf_if_alias
// CHECK: gpu.launch {{.*}} threads([[ALIAS_TID_X:%[^,]+]], [[ALIAS_TID_Y:%[^,]+]],
// CHECK-NOT: arith.cmpi eq, [[ALIAS_TID_Y]]
// CHECK: arith.cmpi eq, [[ALIAS_TID_X]]
// CHECK: acc.atomic.update
func.func @worker_combine_scf_if_alias(
    %result: memref<i32>, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %result_arg = %result,
            %condition_arg = %condition)
        : (!acc.private_type<memref<i32>>, memref<i32>, i1) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c0_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %cast = memref.cast %local
            : memref<i32> to memref<i32, strided<[], offset: ?>>
        %selected = scf.if %condition_arg
            -> (memref<i32, strided<[], offset: ?>>) {
          scf.yield %cast : memref<i32, strided<[], offset: ?>>
        } else {
          scf.yield %cast : memref<i32, strided<[], offset: ?>>
        }
        %result_cast = memref.cast %result_arg
            : memref<i32> to memref<i32, strided<[], offset: ?>>
        acc.predicate_region {
          acc.reduction_combine %selected into %result_cast <add>
              : memref<i32, strided<[], offset: ?>>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
