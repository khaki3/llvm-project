// RUN: fir-opt --cuda-heap-alloc-promotion %s | FileCheck %s --check-prefix=HEAP
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s --check-prefix=LLVM

// Only the managed allocator is device accessible here, so -fstack-arrays
// cannot keep the automatic arrays on the stack.
// RUN: fir-opt --cuda-heap-alloc-promotion=stack-arrays=true %s | FileCheck %s --check-prefix=HEAP

// Same routing as cuda-heap-alloc-unified.fir, with the managed entry points.

// Declarations are emitted at the top of the module, before any function.
// LLVM-DAG: llvm.func @malloc_managed(i64) -> !llvm.ptr
// LLVM-DAG: llvm.func @free_managed(!llvm.ptr)
// LLVM-DAG: llvm.func @malloc(i64) -> !llvm.ptr
// LLVM-DAG: llvm.func @free(!llvm.ptr)

module attributes {fir.cuda_heap_alloc = "managed"} {

// HEAP-LABEL: func.func @vla(
// HEAP: %[[MEM:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "a", fir.cuda_heap_alloc = "managed", fir.must_be_heap = true, uniq_name = "_QFvlaEa"}
// HEAP: fir.freemem %[[MEM]] {fir.cuda_heap_alloc = "managed"} : !fir.heap<!fir.array<?xf32>>
func.func @vla(%arg0: !fir.ref<i32>) {
  %0 = fir.load %arg0 : !fir.ref<i32>
  %1 = fir.convert %0 : (i32) -> index
  %2 = fir.alloca !fir.array<?xf32>, %1 {bindc_name = "a", uniq_name = "_QFvlaEa"}
  %3 = acc.use_device varPtr(%2 : !fir.ref<!fir.array<?xf32>>) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.host_data dataOperands(%3 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// A local the device never sees the host address of stays on the stack.
// HEAP-LABEL: func.func @vla_host_only(
// HEAP: fir.alloca !fir.array<?xf32>
// HEAP-NOT: fir.allocmem
func.func @vla_host_only(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFvla_host_onlyEa"}
  return
}

// HEAP-LABEL: func.func @marked_heap(
// LLVM-LABEL: llvm.func @marked_heap(
// LLVM: llvm.call @malloc_managed(
// LLVM: llvm.call @free_managed(
func.func @marked_heap(%n: index) {
  %0 = fir.allocmem !fir.array<?xf32>, %n {fir.cuda_heap_alloc = "managed"}
  fir.freemem %0 {fir.cuda_heap_alloc = "managed"} : !fir.heap<!fir.array<?xf32>>
  return
}

// LLVM-LABEL: llvm.func @unmarked_heap(
// LLVM-NOT: llvm.call @malloc_managed(
// LLVM: llvm.call @malloc(
// LLVM-NOT: llvm.call @free_managed(
// LLVM: llvm.call @free(
func.func @unmarked_heap(%n: index) {
  %0 = fir.allocmem !fir.array<?xf32>, %n
  fir.freemem %0 : !fir.heap<!fir.array<?xf32>>
  return
}

}
