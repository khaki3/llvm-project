// RUN: fir-opt --cuda-heap-alloc-promotion %s | FileCheck %s --check-prefix=STACK
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s --check-prefix=LLVM
// RUN: fir-opt --fir-to-llvm-ir=unified-heap-alloc-suffix=_pool %s | FileCheck %s --check-prefix=SUFFIX

// fir.cuda_heap_alloc = "unified" leaves the automatic variables alone: the mode
// requires a platform on which the device reaches pageable memory, so they are
// device accessible on the stack, and moving them to the managed allocator only
// costs, because the OpenACC runtime skips the explicit copy for a managed
// pointer and the kernel faults the pages in instead.
// STACK-NOT: fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name
// STACK-NOT: fir.allocmem !fir.char

// The mode still names the allocator of the pairs that carry it, which is what
// an explicitly marked allocation gets.

// Declarations are emitted at the top of the module, before any function.
// LLVM-DAG: llvm.func @malloc_unified(i64) -> !llvm.ptr
// LLVM-DAG: llvm.func @free_unified(!llvm.ptr)
// LLVM-DAG: llvm.func @malloc(i64) -> !llvm.ptr
// LLVM-DAG: llvm.func @free(!llvm.ptr)

module attributes {fir.cuda_heap_alloc = "unified"} {

// STACK-LABEL: func.func @vla(
// STACK: fir.alloca !fir.array<?xf32>
func.func @vla(%arg0: !fir.ref<i32>) {
  %0 = fir.load %arg0 : !fir.ref<i32>
  %1 = fir.convert %0 : (i32) -> index
  %2 = fir.alloca !fir.array<?xf32>, %1 {bindc_name = "a", uniq_name = "_QFvlaEa"}
  return
}

// STACK-LABEL: func.func @autochar(
// STACK: fir.alloca !fir.char<1,?>
func.func @autochar(%arg0: index) {
  %0 = fir.alloca !fir.char<1,?>(%arg0 : index) {bindc_name = "s", uniq_name = "_QFautocharEs"}
  return
}

// LLVM-LABEL: llvm.func @marked_heap(
// LLVM: llvm.call @malloc_unified(
// LLVM: llvm.call @free_unified(
// The entry point names are the libc name plus a configurable suffix.
// SUFFIX-LABEL: llvm.func @marked_heap(
// SUFFIX: llvm.call @malloc_pool(
// SUFFIX: llvm.call @free_pool(
func.func @marked_heap(%n: index) {
  %0 = fir.allocmem !fir.array<?xf32>, %n {fir.cuda_heap_alloc = "unified"}
  fir.freemem %0 {fir.cuda_heap_alloc = "unified"} : !fir.heap<!fir.array<?xf32>>
  return
}

// An unmarked pair belongs to libc: this is the shape of a buffer the Fortran
// runtime allocated and lowered code releases.
// LLVM-LABEL: llvm.func @unmarked_heap(
// LLVM-NOT: llvm.call @malloc_unified(
// LLVM: llvm.call @malloc(
// LLVM-NOT: llvm.call @free_unified(
// LLVM: llvm.call @free(
func.func @unmarked_heap(%n: index) {
  %0 = fir.allocmem !fir.array<?xf32>, %n
  fir.freemem %0 : !fir.heap<!fir.array<?xf32>>
  return
}

}
