// RUN: fir-opt --cuda-heap-alloc-promotion %s | FileCheck %s --check-prefix=HEAP
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s --check-prefix=LLVM
// RUN: fir-opt --fir-to-llvm-ir=unified-heap-alloc-suffix=_pool %s | FileCheck %s --check-prefix=SUFFIX

// Whichever array placement pass follows, the pairs keep their allocator: they
// are marked fir.must_be_heap, so none of those passes moves them back to the
// stack.
// RUN: fir-opt --cuda-heap-alloc-promotion --memory-allocation-opt %s | FileCheck %s --check-prefix=KEEP
// RUN: fir-opt --cuda-heap-alloc-promotion --stack-arrays %s | FileCheck %s --check-prefix=KEEP
// RUN: fir-opt --cuda-heap-alloc-promotion --allocation-placement %s | FileCheck %s --check-prefix=KEEP

// The stack is device accessible here, so -fstack-arrays can keep the automatic
// arrays on it.
// RUN: fir-opt --cuda-heap-alloc-promotion=stack-arrays=true %s | FileCheck %s --check-prefix=STACK
// STACK-NOT: fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name
// STACK-NOT: fir.allocmem !fir.char

// Under fir.cuda_heap_alloc = "unified", a named automatic array whose host
// address is handed to device code moves to the heap and is marked. Only marked
// allocations use malloc_unified: memory the Fortran runtime allocated must
// keep being released by libc free.

// Declarations are emitted at the top of the module, before any function.
// LLVM-DAG: llvm.func @malloc_unified(i64) -> !llvm.ptr
// LLVM-DAG: llvm.func @free_unified(!llvm.ptr)
// LLVM-DAG: llvm.func @malloc(i64) -> !llvm.ptr
// LLVM-DAG: llvm.func @free(!llvm.ptr)

module attributes {fir.cuda_heap_alloc = "unified"} {

// HEAP-LABEL: func.func @vla(
// HEAP: %[[MEM:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "a", fir.cuda_heap_alloc = "unified", fir.must_be_heap = true, uniq_name = "_QFvlaEa"}
// HEAP: fir.freemem %[[MEM]] {fir.cuda_heap_alloc = "unified"} : !fir.heap<!fir.array<?xf32>>
// KEEP-LABEL: func.func @vla(
// KEEP: %[[KMEM:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "a", fir.cuda_heap_alloc = "unified", fir.must_be_heap = true, uniq_name = "_QFvlaEa"}
// KEEP: fir.freemem %[[KMEM]] {fir.cuda_heap_alloc = "unified"} : !fir.heap<!fir.array<?xf32>>
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

// A local the device never sees the host address of stays on the stack: a data
// clause maps and copies it, so host memory is enough.
// HEAP-LABEL: func.func @vla_host_only(
// HEAP: fir.alloca !fir.array<?xf32>
// HEAP-NOT: fir.allocmem
func.func @vla_host_only(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFvla_host_onlyEa"}
  return
}

// Automatic character is an automatic too.
// HEAP-LABEL: func.func @autochar(
// HEAP: fir.allocmem !fir.char<1,?>(%{{.*}} : index) {{{.*}}fir.cuda_heap_alloc = "unified"
func.func @autochar(%arg0: index) {
  %0 = fir.alloca !fir.char<1,?>(%arg0 : index) {bindc_name = "s", uniq_name = "_QFautocharEs"}
  %1 = acc.use_device varPtr(%0 : !fir.ref<!fir.char<1,?>>) name("s") -> !fir.ref<!fir.char<1,?>>
  acc.host_data dataOperands(%1 : !fir.ref<!fir.char<1,?>>) {
    acc.terminator
  }
  return
}

// Fixed-size automatics stay on the stack.
// HEAP-LABEL: func.func @fixed(
// HEAP: fir.alloca !fir.array<128xf32>
// HEAP-NOT: fir.allocmem
func.func @fixed() {
  %0 = fir.alloca !fir.array<128xf32> {bindc_name = "a", uniq_name = "_QFfixedEa"}
  %1 = acc.use_device varPtr(%0 : !fir.ref<!fir.array<128xf32>>) name("a") -> !fir.ref<!fir.array<128xf32>>
  acc.host_data dataOperands(%1 : !fir.ref<!fir.array<128xf32>>) {
    acc.terminator
  }
  return
}

// Compiler temporaries are not automatics: turning them into malloc/free would
// cost an allocation per loop iteration.
// HEAP-LABEL: func.func @anon_temp(
// HEAP: fir.alloca !fir.array<?xf32>
// HEAP-NOT: fir.allocmem
func.func @anon_temp(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0
  %1 = acc.use_device varPtr(%0 : !fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  acc.host_data dataOperands(%1 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// Device code keeps its stack allocation: the entry points are host-only.
// HEAP-LABEL: func.func @device_vla(
// HEAP: fir.alloca !fir.array<?xf32>
// HEAP-NOT: fir.allocmem
func.func @device_vla(%arg0: index) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFdevice_vlaEa"}
  %1 = acc.use_device varPtr(%0 : !fir.ref<!fir.array<?xf32>>) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.host_data dataOperands(%1 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// attributes(host,device) is the host copy of the routine here, so it needs the
// unified memory just like any other host code. The device copy of it lives in
// the gpu.module.
// HEAP-LABEL: func.func @host_device_vla(
// HEAP: fir.allocmem !fir.array<?xf32>, %{{.*}} {{{.*}}fir.cuda_heap_alloc = "unified"
func.func @host_device_vla(%arg0: index) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "a", uniq_name = "_QFhost_device_vlaEa"}
  %1 = acc.use_device varPtr(%0 : !fir.ref<!fir.array<?xf32>>) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.host_data dataOperands(%1 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// An alloca pinned to the stack stays there: the array function result below is
// replaced by the caller buffer, so a heap pair would only be dead code.
// HEAP-LABEL: func.func @array_result(
// HEAP: fir.alloca !fir.array<?xf32>
// HEAP-NOT: fir.allocmem
func.func @array_result(%arg0: index) {
  %0 = fir.alloca !fir.array<?xf32>, %arg0 {bindc_name = "res", fir.must_be_stack = true, uniq_name = "_QFarray_resultEres"}
  %1 = acc.use_device varPtr(%0 : !fir.ref<!fir.array<?xf32>>) name("res") -> !fir.ref<!fir.array<?xf32>>
  acc.host_data dataOperands(%1 : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// HEAP-LABEL: func.func @marked_heap(
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
