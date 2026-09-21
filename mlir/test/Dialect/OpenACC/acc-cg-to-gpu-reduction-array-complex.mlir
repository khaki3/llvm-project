// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// Neither gpu.all_reduce nor memref.atomic_rmw accepts a complex operand, so a
// complex array accumulator must be reduced one component at a time.

// A per-thread complex array accumulator reduces each element with one
// gpu.all_reduce per component, recombined into a complex before the store.

// CHECK-LABEL: func.func @complex_array_reduction
// CHECK: gpu.launch
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xcomplex<f64>>
// CHECK-NOT: acc.reduction_accumulate_array
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[ELT:.*]] = memref.load %[[ALLOCA]][%[[IV]]] : memref<2xcomplex<f64>>
// CHECK:   %[[RE:.*]] = complex.re %[[ELT]]
// CHECK:   %[[RERED:.*]] = gpu.all_reduce add %[[RE]]
// CHECK:   %[[IM:.*]] = complex.im %[[ELT]]
// CHECK:   %[[IMRED:.*]] = gpu.all_reduce add %[[IM]]
// CHECK:   %[[RED:.*]] = complex.create %[[RERED]], %[[IMRED]]
// CHECK:   memref.store %[[RED]], %[[ALLOCA]][%[[IV]]] : memref<2xcomplex<f64>>
// CHECK: }

func.func @complex_array_reduction(%arg0: memref<2xcomplex<f64>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<2xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<2xcomplex<f64>>
  acc.kernel_environment dataOperands(%0 : memref<2xcomplex<f64>>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<2xcomplex<f64>>) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f64
      %one = arith.constant 1.0 : f64
      %czero = complex.create %zero, %zero : complex<f64>
      %cone = complex.create %one, %one : complex<f64>
      %2 = acc.reduction_init %arg2 <add> : memref<2xcomplex<f64>> {
        %alloca = memref.alloca() : memref<2xcomplex<f64>>
        scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
          memref.store %czero, %alloca[%i] : memref<2xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloca : memref<2xcomplex<f64>>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %3 = memref.load %2[%c0] : memref<2xcomplex<f64>>
          %4 = complex.add %3, %cone : complex<f64>
          memref.store %4, %2[%c0] : memref<2xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c2 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> par_dims(#acc<par_dims[block_x, thread_x]>) : memref<2xcomplex<f64>>
      acc.reduction_combine_region %2 into %arg2 : memref<2xcomplex<f64>> {
        scf.for %i = %c0 to %c2 step %c1 {
          %3 = memref.load %2[%i] : memref<2xcomplex<f64>>
          %4 = memref.load %arg2[%i] : memref<2xcomplex<f64>>
          %5 = complex.add %3, %4 : complex<f64>
          memref.store %5, %arg2[%i] : memref<2xcomplex<f64>>
        }
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<2xcomplex<f64>>) to varPtr(%arg0 : memref<2xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// A gang-scoped complex array private_local feeding a worker-level accumulate
// is block-shared, so the in-place update must become an atomic. The body is
// split into real and imaginary updates so the atomic lowering can use two
// scalar atomicrmw instead of a 128-bit compare-exchange.

// CHECK-LABEL: func.func @complex_array_reduction_gang_storage_thread_accum
// CHECK: gpu.launch
// CHECK-NOT: memref.atomic_rmw
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
// CHECK: acc.atomic.update %{{.*}} : memref<1xcomplex<f64>{{.*}}> {
// CHECK: ^bb0(%[[CUR:.*]]: complex<f64>):
// CHECK:   %[[CURRE:.*]] = complex.re %[[CUR]]
// CHECK:   arith.addf %[[CURRE]], %{{.*}} : f64
// CHECK:   %[[CURIM:.*]] = complex.im %[[CUR]]
// CHECK:   arith.addf %[[CURIM]], %{{.*}} : f64
// CHECK:   complex.create
// CHECK:   acc.yield
// The combine reads the block partial from one thread, so the updates must be
// separated from it by a barrier.
// CHECK: gpu.barrier

func.func @complex_array_reduction_gang_storage_thread_accum(%arg0: memref<4xcomplex<f64>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xcomplex<f64>>
  acc.kernel_environment dataOperands(%0 : memref<4xcomplex<f64>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c2 par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x]>) : () -> !acc.private_type<memref<4xcomplex<f64>>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xcomplex<f64>>, !acc.private_type<memref<4xcomplex<f64>>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4_idx = arith.constant 4 : index
      %zero = arith.constant 0.0 : f64
      %one = arith.constant 1.0 : f64
      %czero = complex.create %zero, %zero : complex<f64>
      %cone = complex.create %one, %one : complex<f64>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %local = acc.private_local %priv {acc.par_dims = #acc<par_dims[block_x]>} : (!acc.private_type<memref<4xcomplex<f64>>>) -> memref<4xcomplex<f64>>
        scf.for %i = %c0 to %c4_idx step %c1 {
          memref.store %czero, %local[%i] : memref<4xcomplex<f64>>
        }
        scf.parallel (%wy_iv) = (%c0) to (%kwy) step (%c1) {
          %3 = memref.load %local[%c0] : memref<4xcomplex<f64>>
          %4 = complex.add %3, %cone : complex<f64>
          memref.store %4, %local[%c0] : memref<4xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %b = acc.bounds extent(%c4_idx : index)
        acc.reduction_accumulate_array %local bounds(%b) <add> par_dims(#acc<par_dims[thread_y]>) : memref<4xcomplex<f64>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xcomplex<f64>>) to varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// complex<f32> packs into a single 64-bit compare-exchange rather than two
// component atomics. That is still a correct reduction, so it must lower, not
// report NYI.

// CHECK-LABEL: func.func @complex_f32_array_reduction_shared
// CHECK-NOT: memref.atomic_rmw
// CHECK: acc.atomic.update
// CHECK: gpu.barrier

func.func @complex_f32_array_reduction_shared(%arg0: memref<4xcomplex<f32>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xcomplex<f32>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xcomplex<f32>>
  acc.kernel_environment dataOperands(%0 : memref<4xcomplex<f32>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c2 par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x]>) : () -> !acc.private_type<memref<4xcomplex<f32>>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xcomplex<f32>>, !acc.private_type<memref<4xcomplex<f32>>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4_idx = arith.constant 4 : index
      %zero = arith.constant 0.0 : f32
      %one = arith.constant 1.0 : f32
      %czero = complex.create %zero, %zero : complex<f32>
      %cone = complex.create %one, %one : complex<f32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %local = acc.private_local %priv {acc.par_dims = #acc<par_dims[block_x]>} : (!acc.private_type<memref<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
        scf.for %i = %c0 to %c4_idx step %c1 {
          memref.store %czero, %local[%i] : memref<4xcomplex<f32>>
        }
        scf.parallel (%wy_iv) = (%c0) to (%kwy) step (%c1) {
          %3 = memref.load %local[%c0] : memref<4xcomplex<f32>>
          %4 = complex.add %3, %cone : complex<f32>
          memref.store %4, %local[%c0] : memref<4xcomplex<f32>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %b = acc.bounds extent(%c4_idx : index)
        acc.reduction_accumulate_array %local bounds(%b) <add> par_dims(#acc<par_dims[thread_y]>) : memref<4xcomplex<f32>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xcomplex<f32>>) to varPtr(%arg0 : memref<4xcomplex<f32>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// A predicate region is entered by one thread. The reconvergence barrier for
// the block-shared accumulator must be hoisted ahead of it - a workgroup
// barrier reached by a single thread deadlocks the launch.

// CHECK-LABEL: func.func @complex_array_reduction_predicated
// The barrier sits at the kernel body level, after the update and ahead of the
// predicate, never inside it.
// CHECK: acc.atomic.update
// CHECK: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT: }

func.func @complex_array_reduction_predicated(%arg0: memref<4xcomplex<f64>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xcomplex<f64>>
  acc.kernel_environment dataOperands(%0 : memref<4xcomplex<f64>>) {
    %c1_pw = arith.constant 1 : index
    %c4w = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4w par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0) : (memref<4xcomplex<f64>>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f64
      %one = arith.constant 1.0 : f64
      %czero = complex.create %zero, %zero : complex<f64>
      %cone = complex.create %one, %one : complex<f64>
      %2 = acc.reduction_init %arg2 <add> : memref<4xcomplex<f64>> {
        %alloc = memref.alloc() : memref<4xcomplex<f64>>
        scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
          memref.store %czero, %alloc[%i] : memref<4xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloc : memref<4xcomplex<f64>>
      }
      scf.parallel (%wy_iv) = (%c0) to (%kwy) step (%c1) {
        %3 = memref.load %2[%c0] : memref<4xcomplex<f64>>
        %4 = complex.add %3, %cone : complex<f64>
        memref.store %4, %2[%c0] : memref<4xcomplex<f64>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_y]>}
      acc.predicate_region {
        %b = acc.bounds extent(%c4 : index)
        acc.reduction_accumulate_array %2 bounds(%b) <add> par_dims(#acc<par_dims[block_x, thread_y]>) : memref<4xcomplex<f64>>
      }
      acc.reduction_combine_region %2 into %arg2 : memref<4xcomplex<f64>> {
        scf.for %i = %c0 to %c4 step %c1 {
          %3 = memref.load %2[%i] : memref<4xcomplex<f64>>
          %4 = memref.load %arg2[%i] : memref<4xcomplex<f64>>
          %5 = complex.add %3, %4 : complex<f64>
          memref.store %5, %arg2[%i] : memref<4xcomplex<f64>>
        }
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xcomplex<f64>>) to varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// A loop between the accumulate and its predicate must not stop the search for
// the enclosing predicate, or the barrier stays inside it and deadlocks.

// CHECK-LABEL: func.func @complex_array_reduction_predicated_nested
// CHECK: acc.atomic.update
// CHECK: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT: scf.for

func.func @complex_array_reduction_predicated_nested(%arg0: memref<4xcomplex<f64>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xcomplex<f64>>
  acc.kernel_environment dataOperands(%0 : memref<4xcomplex<f64>>) {
    %c1_pw = arith.constant 1 : index
    %c4w = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4w par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0) : (memref<4xcomplex<f64>>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f64
      %one = arith.constant 1.0 : f64
      %czero = complex.create %zero, %zero : complex<f64>
      %cone = complex.create %one, %one : complex<f64>
      %2 = acc.reduction_init %arg2 <add> : memref<4xcomplex<f64>> {
        %alloc = memref.alloc() : memref<4xcomplex<f64>>
        scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
          memref.store %czero, %alloc[%i] : memref<4xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloc : memref<4xcomplex<f64>>
      }
      scf.parallel (%wy_iv) = (%c0) to (%kwy) step (%c1) {
        %3 = memref.load %2[%c0] : memref<4xcomplex<f64>>
        %4 = complex.add %3, %cone : complex<f64>
        memref.store %4, %2[%c0] : memref<4xcomplex<f64>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_y]>}
      acc.predicate_region {
        scf.for %k = %c0 to %c1 step %c1 {
          %b = acc.bounds extent(%c4 : index)
          acc.reduction_accumulate_array %2 bounds(%b) <add> par_dims(#acc<par_dims[block_x, thread_y]>) : memref<4xcomplex<f64>>
        }
      }
      acc.reduction_combine_region %2 into %arg2 : memref<4xcomplex<f64>> {
        scf.for %i = %c0 to %c4 step %c1 {
          %3 = memref.load %2[%i] : memref<4xcomplex<f64>>
          %4 = memref.load %arg2[%i] : memref<4xcomplex<f64>>
          %5 = complex.add %3, %4 : complex<f64>
          memref.store %5, %arg2[%i] : memref<4xcomplex<f64>>
        }
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xcomplex<f64>>) to varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// A non-additive complex update cannot be split per component, so it compare
// exchanges the packed value. That works while the pair fits in 64 bits.

// CHECK-LABEL: func.func @complex_f32_array_reduction_shared_mul
// CHECK-NOT: memref.atomic_rmw
// CHECK: acc.atomic.update
// CHECK: complex.mul
// CHECK: gpu.barrier

func.func @complex_f32_array_reduction_shared_mul(%arg0: memref<4xcomplex<f32>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xcomplex<f32>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xcomplex<f32>>
  acc.kernel_environment dataOperands(%0 : memref<4xcomplex<f32>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c2 par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x]>) : () -> !acc.private_type<memref<4xcomplex<f32>>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xcomplex<f32>>, !acc.private_type<memref<4xcomplex<f32>>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4_idx = arith.constant 4 : index
      %zero = arith.constant 0.0 : f32
      %one = arith.constant 1.0 : f32
      %czero = complex.create %zero, %zero : complex<f32>
      %cone = complex.create %one, %one : complex<f32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %local = acc.private_local %priv {acc.par_dims = #acc<par_dims[block_x]>} : (!acc.private_type<memref<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
        scf.for %i = %c0 to %c4_idx step %c1 {
          memref.store %czero, %local[%i] : memref<4xcomplex<f32>>
        }
        scf.parallel (%wy_iv) = (%c0) to (%kwy) step (%c1) {
          %3 = memref.load %local[%c0] : memref<4xcomplex<f32>>
          %4 = complex.mul %3, %cone : complex<f32>
          memref.store %4, %local[%c0] : memref<4xcomplex<f32>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %b = acc.bounds extent(%c4_idx : index)
        acc.reduction_accumulate_array %local bounds(%b) <mul> par_dims(#acc<par_dims[thread_y]>) : memref<4xcomplex<f32>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xcomplex<f32>>) to varPtr(%arg0 : memref<4xcomplex<f32>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}
