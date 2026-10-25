// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// The variable the partials are combined into is a dummy argument declared in
// the kernel. The launch that combines them after it declares the variable
// again, without the scope of the dummy arguments of the kernel.

// CHECK-LABEL: func.func @combine_into_declared_dummy
// CHECK:       gpu.launch
// CHECK:         %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:         fir.declare %{{.*}} dummy_scope %[[SCOPE]] arg 2 uniq_name("_QFEr")
// CHECK:         gpu.terminator
// CHECK:       gpu.launch
// CHECK-NOT:     fir.dummy_scope
// CHECK:         %[[DECL:.*]] = fir.declare %{{.*}} uniq_name("_QFEr") : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:         %[[DEST:.*]] = fir.convert %[[DECL]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:         scf.if
// CHECK:           %[[OLD:.*]] = memref.load %[[DEST]][] : memref<i32>
// CHECK:           %[[NEW:.*]] = arith.addi %{{.*}}, %[[OLD]] : i32
// CHECK:           memref.store %[[NEW]], %[[DEST]][] : memref<i32>
// CHECK:         gpu.terminator
// CHECK:       gpu.dealloc
func.func @combine_into_declared_dummy(%arg_in: memref<3xi32>, %arg_res: !fir.ref<i32>, %nb: index) {
  %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
  acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, !fir.ref<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %scope = fir.dummy_scope : !fir.dscope
    %decl = fir.declare %a_res dummy_scope %scope arg 2 uniq_name("_QFEr") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %dest = fir.convert %decl : (!fir.ref<i32>) -> memref<i32>
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
      acc.reduction_combine %acc into %dest <add> par_dims(#acc<par_dims[block_x]>) : memref<i32>
    }
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}
