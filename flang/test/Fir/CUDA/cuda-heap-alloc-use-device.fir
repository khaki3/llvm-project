// RUN: fir-opt --cuda-heap-alloc-promotion %s | FileCheck %s

// What decides the promotion is the host address reaching device code, and
// lowering puts fir.declare and fir.embox in between, so the search has to walk
// through them. Kept apart from cuda-heap-alloc-unified.fir because neither op
// can be taken straight to LLVM IR.

module attributes {fir.cuda_heap_alloc = "unified"} {

// CHECK-LABEL: func.func @host_data(
// CHECK: fir.allocmem !fir.array<?xf32>, %{{.*}} {{{.*}}fir.cuda_heap_alloc = "unified"
// CHECK: fir.freemem %{{.*}} {fir.cuda_heap_alloc = "unified"}
func.func @host_data(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFhost_dataEa"}
  %1 = fir.shape %arg0 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QFhost_dataEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  %3 = fir.embox %2(%1) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %4 = acc.use_device var(%3 : !fir.box<!fir.array<?xf32>>) name("a") -> !fir.box<!fir.array<?xf32>>
  acc.host_data dataOperands(%4 : !fir.box<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// deviceptr hands out the address the same way.
// CHECK-LABEL: func.func @deviceptr(
// CHECK: fir.allocmem !fir.array<?xf32>, %{{.*}} {{{.*}}fir.cuda_heap_alloc = "unified"
func.func @deviceptr(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFdeviceptrEa"}
  %1 = fir.shape %arg0 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QFdeviceptrEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  %3 = acc.deviceptr varPtr(%2 : !fir.ref<!fir.array<?xf32>>) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%3 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// A data clause maps and copies the local, so the device never sees its host
// address and it stays on the stack.
// CHECK-LABEL: func.func @copyin(
// CHECK: fir.alloca !fir.array<?xf32>
// CHECK-NOT: fir.allocmem
func.func @copyin(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFcopyinEa"}
  %1 = fir.shape %arg0 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QFcopyinEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  %3 = acc.copyin varPtr(%2 : !fir.ref<!fir.array<?xf32>>) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%3 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

}
