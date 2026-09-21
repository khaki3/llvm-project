// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" -verify-diagnostics

// A complex accumulator too large for the per-thread stack is not classified as
// per-thread, but its storage is still a stack alloca, so an atomic on it would
// reduce nothing. Report it rather than lowering it or asserting.

func.func @complex_array_reduction_thread_private_nyi(%arg0: memref<4096xcomplex<f64>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4096xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4096xcomplex<f64>>
  acc.kernel_environment dataOperands(%0 : memref<4096xcomplex<f64>>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<4096xcomplex<f64>>) {
      %c2 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f64
      %one = arith.constant 1.0 : f64
      %czero = complex.create %zero, %zero : complex<f64>
      %cone = complex.create %one, %one : complex<f64>
      %2 = acc.reduction_init %arg2 <add> : memref<4096xcomplex<f64>> {
        %alloca = memref.alloca() : memref<4096xcomplex<f64>>
        scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
          memref.store %czero, %alloca[%i] : memref<4096xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloca : memref<4096xcomplex<f64>>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %3 = memref.load %2[%c0] : memref<4096xcomplex<f64>>
          %4 = complex.add %3, %cone : complex<f64>
          // expected-error@+1 {{not yet implemented: reduction: complex array reduction on thread-private storage}}
          memref.store %4, %2[%c0] : memref<4096xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c2 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> par_dims(#acc<par_dims[block_x, thread_x]>) : memref<4096xcomplex<f64>>
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4096xcomplex<f64>>) to varPtr(%arg0 : memref<4096xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// The same update at double precision would need a 128-bit compare exchange
// the target has no instruction for.

func.func @complex_f64_array_reduction_shared_mul(%arg0: memref<4xcomplex<f64>>) {
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
          // expected-error@+2 {{not yet implemented: reduction: unsupported complex reduction operator}}
          %4 = complex.mul %3, %cone : complex<f64>
          memref.store %4, %local[%c0] : memref<4xcomplex<f64>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %b = acc.bounds extent(%c4_idx : index)
        acc.reduction_accumulate_array %local bounds(%b) <mul> par_dims(#acc<par_dims[thread_y]>) : memref<4xcomplex<f64>>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xcomplex<f64>>) to varPtr(%arg0 : memref<4xcomplex<f64>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}
