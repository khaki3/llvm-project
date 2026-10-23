// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// Guard: a gang-only launch keeps combining the partial of each block into
// the shared variable with an atomic, with no thread to predicate.

// CHECK-LABEL: func.func @gang_reduction_combine
// CHECK-SAME:    %{{.*}}: memref<3xi32>, %[[RES:.*]]: memref<i32>
// CHECK:       gpu.launch
// CHECK-NOT:     memref.store %{{.*}}, %[[RES]]
// CHECK:         acc.atomic.update %[[RES]] : memref<i32> {
// CHECK-NEXT:    ^bb0(%[[ARG:.*]]: i32):
// CHECK-NEXT:      arith.addi %{{.*}}, %[[ARG]]
// CHECK-NOT:     memref.store %{{.*}}, %[[RES]]
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_combine_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_combine(%arg_in: memref<3xi32>, %arg_res: memref<i32>) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32) -> i32 {
          %v = memref.load %a_in[%i] : memref<3xi32>
          scf.reduce(%v : i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            %sum = arith.addi %lhs, %rhs : i32
            scf.reduce.return %sum : i32
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        acc.reduction_accumulate %red to %acc <add> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.predicate_region {
        acc.reduction_combine %acc into %a_res <add> par_dims(#acc<par_dims[block_x]>) : memref<i32>
      }
      acc.yield
    } <{kernel_func_name = @gang_reduction_combine_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}

// -----

// A plain store stays a store: combining across blocks is the job of an
// explicit acc.reduction_combine, not inferred from where the value came from.

// CHECK-LABEL: func.func @block_result_store
// CHECK-SAME:    %{{.*}}: memref<3xi32>, %[[RES:.*]]: memref<i32>
// CHECK:       gpu.launch
// CHECK-NOT:     acc.atomic.update
// CHECK:         memref.store %{{.*}}, %[[RES]][] : memref<i32>
// CHECK-NOT:     acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @block_result_store_kernel() kernel {
      gpu.return
    }
  }

  func.func @block_result_store(%arg_in: memref<3xi32>, %arg_res: memref<i32>) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32) -> i32 {
          %v = memref.load %a_in[%i] : memref<3xi32>
          scf.reduce(%v : i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            %sum = arith.addi %lhs, %rhs : i32
            scf.reduce.return %sum : i32
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        acc.reduction_accumulate %red to %acc <add> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %r = memref.load %acc[] : memref<i32>
      acc.predicate_region {
        memref.store %r, %a_res[] : memref<i32>
      }
      acc.yield
    } <{kernel_func_name = @block_result_store_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}

// -----

// The partial of each block is combined into one element of an array through
// a view of that element.

// CHECK-LABEL: func.func @gang_reduction_combine_into_element
// CHECK:       gpu.launch
// CHECK:         %[[ELT:.*]] = memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<8xi32> to memref<i32, strided<[], offset: ?>>
// CHECK-NOT:     memref.store %{{.*}}, %[[ELT]]
// CHECK:         acc.atomic.update %[[ELT]] : memref<i32, strided<[], offset: ?>> {
// CHECK:         gpu.terminator

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_combine_into_element_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_combine_into_element(%arg_in: memref<3xi32>, %arg_res: memref<8xi32>, %arg_k: index) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res, %k = %arg_k) : (memref<3xi32>, memref<8xi32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32) -> i32 {
          %v = memref.load %a_in[%i] : memref<3xi32>
          scf.reduce(%v : i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            %sum = arith.addi %lhs, %rhs : i32
            scf.reduce.return %sum : i32
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        acc.reduction_accumulate %red to %acc <add> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %elt = memref.subview %a_res[%k] [1] [1] : memref<8xi32> to memref<i32, strided<[], offset: ?>>
      acc.predicate_region {
        acc.reduction_combine %acc into %elt <add> par_dims(#acc<par_dims[block_x]>) : memref<i32, strided<[], offset: ?>>, memref<i32>
      }
      acc.yield
    } <{kernel_func_name = @gang_reduction_combine_into_element_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}

// -----

// A complex partial of each block is combined into one array element through
// a view of that element.

// CHECK-LABEL: func.func @gang_complex_combine_into_element
// CHECK:       gpu.launch
// CHECK:         %[[ELT:.*]] = memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<8xcomplex<f32>> to memref<complex<f32>, strided<[], offset: ?>>
// CHECK:         acc.atomic.update %[[ELT]] : memref<complex<f32>, strided<[], offset: ?>> {
// CHECK:           complex.add
// CHECK:         gpu.terminator

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_complex_combine_into_element_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_complex_combine_into_element(%arg_in: memref<3xcomplex<f32>>, %arg_res: memref<8xcomplex<f32>>, %arg_k: index) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res, %k = %arg_k) : (memref<3xcomplex<f32>>, memref<8xcomplex<f32>>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %zero = arith.constant 0.000000e+00 : f32
      %czero = complex.create %zero, %zero : complex<f32>
      %acc = memref.alloca() : memref<complex<f32>>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%czero) -> complex<f32> {
          %v = memref.load %a_in[%i] : memref<3xcomplex<f32>>
          scf.reduce(%v : complex<f32>) {
          ^bb0(%lhs: complex<f32>, %rhs: complex<f32>):
            %sum = complex.add %lhs, %rhs : complex<f32>
            scf.reduce.return %sum : complex<f32>
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        memref.store %red, %acc[] : memref<complex<f32>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %elt = memref.subview %a_res[%k] [1] [1] : memref<8xcomplex<f32>> to memref<complex<f32>, strided<[], offset: ?>>
      acc.predicate_region {
        acc.reduction_combine_region %acc into %elt : memref<complex<f32>>, memref<complex<f32>, strided<[], offset: ?>> {
          %l = memref.load %elt[] : memref<complex<f32>, strided<[], offset: ?>>
          %r = memref.load %acc[] : memref<complex<f32>>
          %s = complex.add %l, %r : complex<f32>
          memref.store %s, %elt[] : memref<complex<f32>, strided<[], offset: ?>>
          acc.yield
        } {acc.par_dims = #acc<par_dims[block_x]>}
      }
      acc.yield
    } <{kernel_func_name = @gang_complex_combine_into_element_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}

// -----

// A combine without block dims outside any loop is applied by one block, but
// with several blocks others may combine into the variable meanwhile, so it is
// atomic; with one block it stays a plain update.

// CHECK-LABEL: func.func @combine_in_multi_block_launch
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK-LABEL: func.func @combine_in_single_block_launch
// CHECK:       gpu.launch
// CHECK-NOT:     acc.atomic.update
// CHECK:         gpu.terminator

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @multi_kernel() kernel {
      gpu.return
    }
    gpu.func @single_kernel() kernel {
      gpu.return
    }
  }

  func.func @combine_in_multi_block_launch(%arg_res: memref<i32>) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_res = %arg_res) : (memref<i32>) {
      %c1_i32 = arith.constant 1 : i32
      %acc = memref.alloca() : memref<i32>
      memref.store %c1_i32, %acc[] : memref<i32>
      acc.predicate_region {
        acc.reduction_combine %acc into %a_res <add> par_dims(#acc<par_dims[thread_x]>) : memref<i32>
      } {acc.active_par_dims = #acc<active_par_dims[]>}
      acc.yield
    } <{kernel_func_name = @multi_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.parallel"}>
    return
  }

  func.func @combine_in_single_block_launch(%arg_res: memref<i32>) {
    %c1 = arith.constant 1 : index
    %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_res = %arg_res) : (memref<i32>) {
      %c1_i32 = arith.constant 1 : i32
      %acc = memref.alloca() : memref<i32>
      memref.store %c1_i32, %acc[] : memref<i32>
      acc.predicate_region {
        acc.reduction_combine %acc into %a_res <add> par_dims(#acc<par_dims[thread_x]>) : memref<i32>
      } {acc.active_par_dims = #acc<active_par_dims[]>}
      acc.yield
    } <{kernel_func_name = @single_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.parallel"}>
    return
  }
}
