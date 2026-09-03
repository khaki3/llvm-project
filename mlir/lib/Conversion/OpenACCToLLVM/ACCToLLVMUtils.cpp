//===- ACCToLLVMUtils.cpp - OpenACC to LLVM helpers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::acc;

Location acc::unfuseLoc(Location loc) {
  while (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    loc = fusedLoc.getLocations().back();
  return loc;
}

/// Peels the acc.loop layer, exposing the [base, inclusion...] chain beneath.
static Location unfuseLoopLoc(Location loc) {
  while (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    auto loopLoc = dyn_cast_if_present<LoopLocAttr>(fusedLoc.getMetadata());
    if (!loopLoc || fusedLoc.getLocations().empty())
      break;
    loc = loopLoc.getDirective() &&
                  !isa<UnknownLoc>(Location(loopLoc.getDirective()))
              ? Location(loopLoc.getDirective())
              : fusedLoc.getLocations().front();
  }
  return loc;
}

Location acc::unfuseBaseLoc(Location loc) {
  // Fused sub-locations start with the base; acc.loop additionally names that
  // base -- its directive -- in the LoopLocAttr.
  while (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    Location next = loc;
    auto loopLoc = dyn_cast_if_present<LoopLocAttr>(fusedLoc.getMetadata());
    if (loopLoc && loopLoc.getDirective() &&
        !isa<UnknownLoc>(Location(loopLoc.getDirective())))
      next = Location(loopLoc.getDirective());
    else if (!fusedLoc.getLocations().empty())
      next = fusedLoc.getLocations().front();
    if (next == loc)
      break;
    loc = next;
  }
  return loc;
}

std::optional<FileLineColLoc>
acc::getFileLineColLoc(Location loc, bool errorOnInvalidLocation) {
  Location unfusedLoc = unfuseLoc(loc);

  if (auto fileLoc = dyn_cast<FileLineColLoc>(unfusedLoc))
    return fileLoc;

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(unfusedLoc)) {
    if (auto calleeFileLoc = getFileLineColLoc(callSiteLoc.getCallee(), false))
      return calleeFileLoc;
    if (auto callerFileLoc =
            getFileLineColLoc(callSiteLoc.getCaller(), errorOnInvalidLocation))
      return callerFileLoc;
  }

  if (errorOnInvalidLocation)
    llvm_unreachable(
        "cannot get file:line information: invalid Location information");
  return std::nullopt;
}

StringRef acc::getParentFunctionName(Operation *op) {
  if (!op)
    return "";
  if (auto parentFuncOp = op->getParentOfType<func::FuncOp>())
    return parentFuncOp.getName();
  if (auto parentFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>())
    return parentFuncOp.getSymName();
  return "";
}

StringRef acc::getParentFunctionName(Value value) {
  if (auto *op = value.getDefiningOp())
    return getParentFunctionName(op);
  return "";
}

StringRef acc::getParentFunctionName(ValueRange values) {
  for (Value value : values) {
    if (StringRef name = getParentFunctionName(value); !name.empty())
      return name;
  }
  return "";
}

/// Creates or reuses a module-internal null-terminated string global and
/// returns the GlobalOp.
static LLVM::GlobalOp getOrCreateGlobalStringOp(Location loc,
                                                OpBuilder &builder,
                                                StringRef name, StringRef value,
                                                ModuleOp module) {
  if (auto global = module.lookupSymbol<LLVM::GlobalOp>(name))
    return global;

  // Materialize the global through the incoming builder so that it stays
  // tracked when the caller is a dialect conversion rewriter.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  SmallString<32> nullTermStr(value);
  nullTermStr.push_back('\0');
  auto arrayTy = LLVM::LLVMArrayType::get(builder.getI8Type(),
                                          nullTermStr.size_in_bytes());
  return LLVM::GlobalOp::create(builder, loc, arrayTy, /*isConstant=*/true,
                                LLVM::Linkage::Internal, name,
                                builder.getStringAttr(nullTermStr),
                                /*alignment=*/0);
}

Value acc::getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                   StringRef name, StringRef value,
                                   ModuleOp module) {
  Type i64Ty = builder.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  LLVM::GlobalOp global =
      getOrCreateGlobalStringOp(loc, builder, name, value, module);

  Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
  Value cst0 = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                        builder.getI64IntegerAttr(0));
  return LLVM::GEPOp::create(builder, loc, ptrTy, global.getType(), globalPtr,
                             ArrayRef<Value>({cst0, cst0}));
}

Value acc::createIdent(Location loc, StringRef functionName, OpBuilder &builder,
                       ModuleOp module, const ACCRuntimeCallConfig &config) {
  MLIRContext *ctx = builder.getContext();
  Type i32Ty = builder.getI32Type();
  Type i64Ty = builder.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, {i32Ty, i32Ty, i32Ty, i32Ty, ptrTy});

  std::string source;
  std::string sourceGlobalName;
  Location baseLoc = unfuseBaseLoc(loc);
  if (auto fileLineColLoc =
          getFileLineColLoc(baseLoc, /*errorOnInvalidLocation=*/false)) {
    std::string filename = fileLineColLoc->getFilename().str();
    // The runtime reads the field up to the next ';', so the INCLUDE chain
    // rides along in the filename rather than needing a new ident field.
    if (auto fusedLoc = dyn_cast<FusedLoc>(unfuseLoopLoc(loc));
        fusedLoc && fusedLoc.getMetadata())
      for (Location inclusion : fusedLoc.getLocations().drop_front())
        if (auto incLoc = getFileLineColLoc(inclusion, false))
          filename += " (included from " + incLoc->getFilename().str() + ":" +
                      std::to_string(incLoc->getLine()) + ")";
    std::string line = std::to_string(fileLineColLoc->getLine());
    std::string column = std::to_string(fileLineColLoc->getColumn());
    std::string functionDisplayName =
        functionName.empty() ? std::string()
                             : config.getFunctionDisplayName(functionName);
    source = ";";
    source += filename + ";";
    source += functionDisplayName + ";";
    source += line + ";";
    source += column + ";";
    source += ";";
    sourceGlobalName = "loc_";
    sourceGlobalName += line + "_";
    sourceGlobalName += column + "_";
    sourceGlobalName +=
        std::to_string(static_cast<uint64_t>(llvm::hash_value(source)));
  } else {
    source = ";unknown;unknown;0;0;;";
    sourceGlobalName = "loc__";
  }

  std::string identGlobalName = "ident_";
  identGlobalName += sourceGlobalName;
  auto identGlobal = module.lookupSymbol<LLVM::GlobalOp>(identGlobalName);
  if (!identGlobal) {
    LLVM::GlobalOp sourceGlobal = getOrCreateGlobalStringOp(
        loc, builder, sourceGlobalName, source, module);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(sourceGlobal);
    identGlobal = LLVM::GlobalOp::create(
        builder, loc, structTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        identGlobalName, /*value=*/Attribute(), /*alignment=*/0);

    Block *block = builder.createBlock(&identGlobal.getInitializerRegion());
    builder.setInsertionPointToStart(block);
    Value ident = LLVM::ZeroOp::create(builder, loc, structTy);
    Value sourceBase = LLVM::AddressOfOp::create(builder, loc, sourceGlobal);
    Value cst0 = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                          builder.getI64IntegerAttr(0));
    Value sourcePtr =
        LLVM::GEPOp::create(builder, loc, ptrTy, sourceGlobal.getType(),
                            sourceBase, ArrayRef<Value>({cst0, cst0}));
    ident = LLVM::InsertValueOp::create(builder, loc, structTy, ident,
                                        sourcePtr, ArrayRef<int64_t>{4});
    LLVM::ReturnOp::create(builder, loc, ident);
  }

  return LLVM::AddressOfOp::create(builder, loc, identGlobal);
}
