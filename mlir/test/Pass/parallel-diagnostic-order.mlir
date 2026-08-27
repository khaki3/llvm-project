// Diagnostics emitted from parallel pass execution must appear in the order a
// serial run would produce, including from nested parallel adaptors.

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}), builtin.module(func.func(test-pass-failure{gen-diagnostics})))' -o /dev/null 2>&1 | FileCheck %s
// RUN: not mlir-opt %s --mlir-disable-threading -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}), builtin.module(func.func(test-pass-failure{gen-diagnostics})))' -o /dev/null 2>&1 | FileCheck %s

func.func @f0() { return } loc("fn_00")
func.func @f1() { return } loc("fn_01")
func.func @f2() { return } loc("fn_02")
func.func @f3() { return } loc("fn_03")
func.func @f4() { return } loc("fn_04")
func.func @f5() { return } loc("fn_05")
func.func @f6() { return } loc("fn_06")
func.func @f7() { return } loc("fn_07")
func.func @f8() { return } loc("fn_08")
func.func @f9() { return } loc("fn_09")
func.func @f10() { return } loc("fn_10")
func.func @f11() { return } loc("fn_11")

module @m0 {
  func.func @g0() { return } loc("mod0_fn_00")
  func.func @g1() { return } loc("mod0_fn_01")
  func.func @g2() { return } loc("mod0_fn_02")
}
module @m1 {
  func.func @g0() { return } loc("mod1_fn_00")
  func.func @g1() { return } loc("mod1_fn_01")
  func.func @g2() { return } loc("mod1_fn_02")
}
module @m2 {
  func.func @g0() { return } loc("mod2_fn_00")
  func.func @g1() { return } loc("mod2_fn_01")
  func.func @g2() { return } loc("mod2_fn_02")
}
module @m3 {
  func.func @g0() { return } loc("mod3_fn_00")
  func.func @g1() { return } loc("mod3_fn_01")
  func.func @g2() { return } loc("mod3_fn_02")
}

// CHECK: loc("fn_00")
// CHECK-NEXT: loc("fn_01")
// CHECK-NEXT: loc("fn_02")
// CHECK-NEXT: loc("fn_03")
// CHECK-NEXT: loc("fn_04")
// CHECK-NEXT: loc("fn_05")
// CHECK-NEXT: loc("fn_06")
// CHECK-NEXT: loc("fn_07")
// CHECK-NEXT: loc("fn_08")
// CHECK-NEXT: loc("fn_09")
// CHECK-NEXT: loc("fn_10")
// CHECK-NEXT: loc("fn_11")
// CHECK-NEXT: loc("mod0_fn_00")
// CHECK-NEXT: loc("mod0_fn_01")
// CHECK-NEXT: loc("mod0_fn_02")
// CHECK-NEXT: loc("mod1_fn_00")
// CHECK-NEXT: loc("mod1_fn_01")
// CHECK-NEXT: loc("mod1_fn_02")
// CHECK-NEXT: loc("mod2_fn_00")
// CHECK-NEXT: loc("mod2_fn_01")
// CHECK-NEXT: loc("mod2_fn_02")
// CHECK-NEXT: loc("mod3_fn_00")
// CHECK-NEXT: loc("mod3_fn_01")
// CHECK-NEXT: loc("mod3_fn_02")
