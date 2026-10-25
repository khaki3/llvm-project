// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// The partial of each block is stored to its slot of a buffer. A one-block
// launch after the kernel combines the slots into the variable in a fixed
// order: each thread folds every 256th slot from its own, then thread 0 folds
// the results of the threads.

// CHECK-LABEL: func.func @gang_reduction_combine
// CHECK-SAME:    %{{.*}}: memref<3xi32>, %[[RES:.*]]: memref<i32>, %[[NB:.*]]: index
// CHECK:       %[[NXY:.*]] = arith.muli %[[NB]], %{{.*}} : index
// CHECK-NEXT:  %[[NUM:.*]] = arith.muli %[[NXY]], %{{.*}} : index
// CHECK-NEXT:  %[[PARTIALS:.*]] = gpu.alloc  (%[[NUM]]) : memref<?xi32>
// CHECK:       gpu.launch blocks(%[[BX:[^,]*]], %[[BY:[^,]*]], %[[BZ:[^)]*]]) in (%[[GX:[^ ]*]] = %[[NB]], %[[GY:[^ ]*]] = %{{.*}}, %{{.*}} = %{{.*}})
// CHECK-NOT:     acc.atomic.update
// CHECK:         %[[PARTIAL:.*]] = memref.load %{{.*}}[] : memref<i32>
// CHECK-NEXT:    %[[ZY:.*]] = arith.muli %[[BZ]], %[[GY]] : index
// CHECK-NEXT:    %[[Y:.*]] = arith.addi %[[ZY]], %[[BY]] : index
// CHECK-NEXT:    %[[YX:.*]] = arith.muli %[[Y]], %[[GX]] : index
// CHECK-NEXT:    %[[SLOT:.*]] = arith.addi %[[YX]], %[[BX]] : index
// CHECK-NEXT:    memref.store %[[PARTIAL]], %[[PARTIALS]][%[[SLOT]]] : memref<?xi32>
// CHECK-NEXT:    gpu.terminator
// CHECK:       %[[ONE:.*]] = arith.constant 1 : index
// CHECK:       %[[C256:.*]] = arith.constant 256 : index
// CHECK:       gpu.launch blocks({{.*}}) in ({{.*}} = %[[ONE]], {{.*}} = %[[ONE]], {{.*}} = %[[ONE]]) threads(%[[TID:[^,]*]], {{.*}}) in (%[[WIDTH:[^ ]*]] = %[[C256]], {{.*}} = %[[ONE]], {{.*}} = %[[ONE]])
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         %[[STEP:.*]] = arith.constant 1 : index
// CHECK:         %[[OWN:.*]] = arith.cmpi ult, %[[TID]], %[[NUM]] : index
// CHECK:         scf.if %[[OWN]] {
// CHECK:           %[[NEXT:.*]] = arith.addi %[[TID]], %[[WIDTH]] : index
// CHECK:           %[[FIRST:.*]] = memref.load %[[PARTIALS]][%[[TID]]] : memref<?xi32>
// CHECK:           %[[SHARE:.*]] = scf.for %[[I:.*]] = %[[NEXT]] to %[[NUM]] step %[[WIDTH]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
// CHECK:             %[[P:.*]] = memref.load %[[PARTIALS]][%[[I]]] : memref<?xi32>
// CHECK:             %[[SUM:.*]] = arith.addi %[[ACC]], %[[P]] : i32
// CHECK:             scf.yield %[[SUM]] : i32
// CHECK:           memref.store %[[SHARE]], %[[PARTIALS]][%[[TID]]] : memref<?xi32>
// CHECK:         gpu.barrier
// CHECK:         %[[IS_ZERO:.*]] = arith.cmpi eq, %[[TID]], %[[ZERO]] : index
// CHECK:         %[[IS_FIRST:.*]] = arith.andi %[[OWN]], %[[IS_ZERO]] : i1
// CHECK:         scf.if %[[IS_FIRST]] {
// CHECK:           %[[SHARES:.*]] = arith.minui %[[NUM]], %[[WIDTH]] : index
// CHECK:           %[[FIRST_SHARE:.*]] = memref.load %[[PARTIALS]][%[[ZERO]]] : memref<?xi32>
// CHECK:           %[[TOTAL:.*]] = scf.for %[[J:.*]] = %[[STEP]] to %[[SHARES]] step %[[STEP]] iter_args(%{{.*}} = %[[FIRST_SHARE]]) -> (i32) {
// CHECK:             memref.load %[[PARTIALS]][%[[J]]] : memref<?xi32>
// CHECK:           %[[OLD:.*]] = memref.load %[[RES]][] : memref<i32>
// CHECK:           %[[NEW:.*]] = arith.addi %[[TOTAL]], %[[OLD]] : i32
// CHECK:           memref.store %[[NEW]], %[[RES]][] : memref<i32>
// CHECK:         gpu.terminator
// CHECK:       gpu.dealloc  %[[PARTIALS]] : memref<?xi32>

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_combine_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_combine(%arg_in: memref<3xi32>, %arg_res: memref<i32>, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
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
// a view of that element, made again in the launch after the kernel.

// CHECK-LABEL: func.func @gang_reduction_combine_into_element
// CHECK-SAME:    %[[ARR:[^:]*]]: memref<8xi32>, %[[K:[^:]*]]: index
// CHECK:       gpu.launch
// CHECK-NOT:     acc.atomic.update
// CHECK:         gpu.terminator
// CHECK:       gpu.launch
// CHECK-NEXT:    %[[ELT:.*]] = memref.subview %[[ARR]][%[[K]]] [1] [1] : memref<8xi32> to memref<i32, strided<[], offset: ?>>
// CHECK:         %[[OLD:.*]] = memref.load %[[ELT]][] : memref<i32, strided<[], offset: ?>>
// CHECK-NEXT:    %[[NEW:.*]] = arith.addi %{{.*}}, %[[OLD]] : i32
// CHECK-NEXT:    memref.store %[[NEW]], %[[ELT]][] : memref<i32, strided<[], offset: ?>>

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_combine_into_element_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_combine_into_element(%arg_in: memref<3xi32>, %arg_res: memref<8xi32>, %arg_k: index, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
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
// CHECK:       gpu.alloc  (%{{.*}}) : memref<?xcomplex<f32>>
// CHECK:       gpu.launch
// CHECK-NOT:     acc.atomic.update
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xcomplex<f32>>
// CHECK:       gpu.launch
// CHECK:         %[[ELT:.*]] = memref.subview
// CHECK:         complex.add
// CHECK:         complex.add
// CHECK:         %[[OLD:.*]] = memref.load %[[ELT]][] : memref<complex<f32>, strided<[], offset: ?>>
// CHECK-NEXT:    %[[NEW:.*]] = complex.add %{{.*}}, %[[OLD]] : complex<f32>
// CHECK-NEXT:    memref.store %[[NEW]], %[[ELT]][] : memref<complex<f32>, strided<[], offset: ?>>

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_complex_combine_into_element_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_complex_combine_into_element(%arg_in: memref<3xcomplex<f32>>, %arg_res: memref<8xcomplex<f32>>, %arg_k: index, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
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

// Two combines share the launch after the kernel, each with its own buffer.

// CHECK-LABEL: func.func @two_gang_reduction_combines
// CHECK-SAME:    %{{.*}}: memref<3xi32>, %[[SUM:.*]]: memref<i32>, %[[MAX:.*]]: memref<i32>, %{{.*}}: index
// CHECK:       %[[SUMS:.*]] = gpu.alloc
// CHECK:       %[[MAXES:.*]] = gpu.alloc
// CHECK:       gpu.launch
// CHECK:         memref.store %{{.*}}, %[[SUMS]][%{{.*}}] : memref<?xi32>
// CHECK:         memref.store %{{.*}}, %[[MAXES]][%{{.*}}] : memref<?xi32>
// CHECK:       gpu.launch
// CHECK:         memref.store %{{.*}}, %[[SUM]][] : memref<i32>
// CHECK:         arith.maxsi
// CHECK:         memref.store %{{.*}}, %[[MAX]][] : memref<i32>
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch
// CHECK:       gpu.dealloc  %[[MAXES]]
// CHECK:       gpu.dealloc  %[[SUMS]]

module attributes {gpu.container_module} {
  func.func @two_gang_reduction_combines(%arg_in: memref<3xi32>, %arg_sum: memref<i32>, %arg_max: memref<i32>, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_sum = %arg_sum, %a_max = %arg_max) : (memref<3xi32>, memref<i32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %sum = memref.alloca() : memref<i32>
      %max = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red:2 = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32, %c0_i32) -> (i32, i32) {
          %v = memref.load %a_in[%i] : memref<3xi32>
          scf.reduce(%v, %v : i32, i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            %s = arith.addi %lhs, %rhs : i32
            scf.reduce.return %s : i32
          }, {
          ^bb0(%lhs: i32, %rhs: i32):
            %m = arith.maxsi %lhs, %rhs : i32
            scf.reduce.return %m : i32
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        acc.reduction_accumulate %red#0 to %sum <add> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        acc.reduction_accumulate %red#1 to %max <max> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.predicate_region {
        acc.reduction_combine %sum into %a_sum <add> par_dims(#acc<par_dims[block_x]>) : memref<i32>
        acc.reduction_combine %max into %a_max <max> par_dims(#acc<par_dims[block_x]>) : memref<i32>
      }
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }
}

// -----

// The partials are combined atomically with a single block, ...

// CHECK-LABEL: func.func @gang_reduction_combine_single_block
// CHECK-NOT:   gpu.alloc
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_combine_single_block_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_combine_single_block(%arg_in: memref<3xi32>, %arg_res: memref<i32>) {
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
    } <{kernel_func_name = @gang_reduction_combine_single_block_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}

// -----

// ... with a combine a block runs more than once, ...

// CHECK-LABEL: func.func @gang_reduction_combine_in_loop
// CHECK-NOT:   gpu.alloc
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  func.func @gang_reduction_combine_in_loop(%arg_in: memref<3xi32>, %arg_res: memref<i32>, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.for %j = %c0 to %c3 step %c1 {
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
      }
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }
}

// -----

// ... with an element addressed by a value the kernel loads, ...

// CHECK-LABEL: func.func @gang_reduction_combine_loaded_index
// CHECK-NOT:   gpu.alloc
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  func.func @gang_reduction_combine_loaded_index(%arg_in: memref<3xi32>, %arg_res: memref<8xi32>, %arg_k: memref<index>, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res, %a_k = %arg_k) : (memref<3xi32>, memref<8xi32>, memref<index>) {
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
      %k = memref.load %a_k[] : memref<index>
      %elt = memref.subview %a_res[%k] [1] [1] : memref<8xi32> to memref<i32, strided<[], offset: ?>>
      acc.predicate_region {
        acc.reduction_combine %acc into %elt <add> par_dims(#acc<par_dims[block_x]>) : memref<i32, strided<[], offset: ?>>, memref<i32>
      }
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }
}

// -----

// ... on a stream, ...

// CHECK-LABEL: func.func @gang_reduction_combine_on_stream
// CHECK-NOT:   gpu.alloc
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  func.func @gang_reduction_combine_on_stream(%arg_in: memref<3xi32>, %arg_res: memref<i32>, %stream: !gpu.async.token, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
    acc.compute_region stream(%stream : !gpu.async.token) launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<i32>) {
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
    } <{origin = "acc.parallel"}>
    return
  }
}

// -----

// ... with a partial in each worker row of a block, ...

// CHECK-LABEL: func.func @worker_rows_combine
// CHECK-NOT:   gpu.alloc
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  func.func @worker_rows_combine(%arg_in: memref<3xi32>, %arg_res: memref<i32>, %nb: index) {
    %c4 = arith.constant 4 : index
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
    %ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    acc.compute_region launch(%kbx = %bx, %kty = %ty) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%ty_iv) = (%c0) to (%kty) step (%c1) {
          %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32) -> i32 {
            %v = memref.load %a_in[%i] : memref<3xi32>
            scf.reduce(%v : i32) {
            ^bb0(%lhs: i32, %rhs: i32):
              %sum = arith.addi %lhs, %rhs : i32
              scf.reduce.return %sum : i32
            }
          } {acc.par_dims = #acc<par_dims[sequential]>}
          memref.store %red, %acc[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.predicate_region {
        acc.reduction_combine %acc into %a_res <add> par_dims(#acc<par_dims[block_x, thread_y]>) : memref<i32>
      } {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }
}

// -----

// ... or in a predicate region inside one that only block 0 runs.

// CHECK-LABEL: func.func @gang_reduction_combine_in_block_zero
// CHECK-NOT:   gpu.alloc
// CHECK:       gpu.launch
// CHECK:         acc.atomic.update
// CHECK:         gpu.terminator
// CHECK-NOT:   gpu.launch

module attributes {gpu.container_module} {
  func.func @gang_reduction_combine_in_block_zero(%arg_in: memref<3xi32>, %arg_res: memref<i32>, %nb: index) {
    %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
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
        acc.predicate_region {
          acc.reduction_combine %acc into %a_res <add> par_dims(#acc<par_dims[block_x]>) : memref<i32>
        }
      } {acc.active_par_dims = #acc<active_par_dims[]>}
      acc.yield
    } <{origin = "acc.parallel"}>
    return
  }
}
