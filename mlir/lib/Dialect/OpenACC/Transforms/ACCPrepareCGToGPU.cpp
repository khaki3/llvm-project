//===- ACCPrepareCGToGPU.cpp - Decide OpenACC GPU policy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::acc {
#define GEN_PASS_DEF_ACCPREPARECGTOGPU
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace mlir::acc

using namespace mlir;
using namespace mlir::acc;

namespace {

struct Ownership {
  ArrayReductionMode mode;
  llvm::SmallSetVector<PrivateLocalOp, 2> privateLocals;
};

static std::optional<Value> getConstantSelectedValue(arith::SelectOp select) {
  Attribute constant;
  if (!matchPattern(select.getCondition(), m_Constant(&constant)))
    return std::nullopt;
  auto condition = dyn_cast<IntegerAttr>(constant);
  if (!condition)
    return std::nullopt;
  return condition.getValue().isOne() ? select.getTrueValue()
                                      : select.getFalseValue();
}

static std::optional<bool> getConstantCondition(Value condition) {
  Attribute constant;
  if (!matchPattern(condition, m_Constant(&constant)))
    return std::nullopt;
  if (auto value = dyn_cast<IntegerAttr>(constant))
    return value.getValue().isOne();
  return std::nullopt;
}

static bool isPassThroughAliasOf(Value value, Value source) {
  llvm::DenseSet<Value> visited;
  while (value != source) {
    if (!visited.insert(value).second)
      return false;

    if (auto view = value.getDefiningOp<ViewLikeOpInterface>()) {
      if (view.getViewDest() != value)
        return false;
      value = view.getViewSource();
      continue;
    }
    if (auto partial = dyn_cast_or_null<PartialEntityAccessOpInterface>(
            value.getDefiningOp())) {
      value = partial.getBaseEntity();
      continue;
    }
    return false;
  }
  return true;
}

static int64_t getElementBytes(Type type) {
  if (type.isIntOrFloat()) {
    unsigned bits = type.getIntOrFloatBitWidth();
    return bits % 8 == 0 ? bits / 8 : 0;
  }
  if (auto complexTy = dyn_cast<ComplexType>(type)) {
    int64_t componentBytes = getElementBytes(complexTy.getElementType());
    return componentBytes > 0 ? 2 * componentBytes : 0;
  }
  return 0;
}

static bool fitsThreadStack(MemRefType type, int64_t maxBytes) {
  if (!type.hasStaticShape())
    return false;
  int64_t bytes = getElementBytes(type.getElementType());
  if (bytes <= 0)
    return false;
  for (int64_t extent : type.getShape()) {
    if (extent < 0 || (extent != 0 && bytes > maxBytes / extent))
      return false;
    bytes *= extent;
  }
  return bytes < maxBytes;
}

class PolicyAnalyzer {
public:
  explicit PolicyAnalyzer(int64_t maxThreadPrivateStack)
      : maxThreadPrivateStack(maxThreadPrivateStack) {}

  FailureOr<Ownership> classify(Value value) {
    if (auto found = cache.find(value); found != cache.end())
      return found->second;
    if (!visiting.insert(value).second)
      return failure();

    FailureOr<Ownership> result = classifyImpl(value);
    visiting.erase(value);
    if (succeeded(result))
      cache.try_emplace(value, *result);
    return result;
  }

  PrivateStorageKind getStorageKind(PrivateLocalOp privateLocal) const {
    auto privateTy = cast<PrivateType>(privateLocal.getPrivatized().getType());
    auto memrefTy = dyn_cast<MemRefType>(privateTy.getBaseTy());
    if (getReductionMode(privateLocal) == ArrayReductionMode::PerThread &&
        memrefTy && fitsThreadStack(memrefTy, maxThreadPrivateStack))
      return PrivateStorageKind::PerThread;
    return PrivateStorageKind::IndexedGlobal;
  }

private:
  ArrayReductionMode getReductionMode(PrivateLocalOp privateLocal) const {
    ComputeRegionOp computeRegion =
        privateLocal->getParentOfType<ComputeRegionOp>();
    PrivatizeOp privatize = getPrivatizeOp(privateLocal, computeRegion);
    GPUParallelDimsAttr dims = acc::getParDimsAttr(privateLocal);
    if (!dims)
      dims = privatize.getParDimsAttr();

    bool hasThreadX = false;
    bool hasThreadY = false;
    if (dims) {
      for (GPUParallelDimAttr dim : dims.getArray()) {
        hasThreadX |= dim.isThreadX();
        hasThreadY |= dim.isThreadY();
      }
    }

    if (hasThreadY && !hasThreadX)
      return ArrayReductionMode::WorkerRow;

    auto privateTy = cast<PrivateType>(privateLocal.getPrivatized().getType());
    auto memrefTy = dyn_cast<MemRefType>(privateTy.getBaseTy());
    if (hasThreadX)
      return memrefTy && fitsThreadStack(memrefTy, maxThreadPrivateStack)
                 ? ArrayReductionMode::PerThread
                 : ArrayReductionMode::BlockSharedNoOp;
    if (!dims && memrefTy && fitsThreadStack(memrefTy, maxThreadPrivateStack))
      return ArrayReductionMode::PerThread;
    return ArrayReductionMode::BlockSharedNoOp;
  }

  FailureOr<Ownership> classifyFor(scf::ForOp forOp, Value requested) {
    SmallVector<Ownership> ownerships;
    ownerships.reserve(forOp.getNumResults());
    for (Value init : forOp.getInitArgs()) {
      FailureOr<Ownership> ownership = classify(init);
      if (failed(ownership))
        return failure();
      ownerships.push_back(*ownership);
    }

    auto iterArgs = forOp.getRegionIterArgs();
    for (auto [iterArg, result, ownership] :
         llvm::zip_equal(iterArgs, forOp.getResults(), ownerships)) {
      cache[iterArg] = ownership;
      cache[result] = ownership;
    }

    auto clearLoopResultCache = [&]() {
      forOp.getBody()->walk([&](Operation *op) {
        for (Value result : op->getResults())
          cache.erase(result);
      });
    };
    auto clearLoopOwnershipCache = [&]() {
      for (auto [iterArg, result] :
           llvm::zip_equal(iterArgs, forOp.getResults())) {
        cache.erase(iterArg);
        cache.erase(result);
      }
    };

    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    bool changed;
    do {
      changed = false;
      clearLoopResultCache();
      for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
        FailureOr<Ownership> yieldedOwnership =
            isPassThroughAliasOf(yielded, iterArgs[index])
                ? FailureOr<Ownership>(ownerships[index])
                : classify(yielded);
        if (failed(yieldedOwnership) ||
            ownerships[index].mode != yieldedOwnership->mode) {
          clearLoopOwnershipCache();
          return failure();
        }

        size_t oldSize = ownerships[index].privateLocals.size();
        ownerships[index].privateLocals.insert(
            yieldedOwnership->privateLocals.begin(),
            yieldedOwnership->privateLocals.end());
        changed |= oldSize != ownerships[index].privateLocals.size();
        cache[iterArgs[index]] = ownerships[index];
        cache[forOp.getResult(index)] = ownerships[index];
      }
    } while (changed);

    for (auto [iterArg, result, ownership] :
         llvm::zip_equal(iterArgs, forOp.getResults(), ownerships)) {
      cache[iterArg] = ownership;
      cache[result] = ownership;
    }
    return cache.lookup(requested);
  }

  FailureOr<Ownership> classifyImpl(Value value) {
    if (auto reductionInit = value.getDefiningOp<ReductionInitOp>()) {
      auto yield =
          cast<acc::YieldOp>(reductionInit.getRegion().front().getTerminator());
      return classify(yield.getOperand(0));
    }
    if (auto privateLocal = value.getDefiningOp<PrivateLocalOp>()) {
      Ownership result{getReductionMode(privateLocal), {}};
      result.privateLocals.insert(privateLocal);
      return result;
    }
    if (value.getDefiningOp<memref::AllocaOp>())
      return Ownership{ArrayReductionMode::PerThread, {}};
    if (value.getDefiningOp<memref::AllocOp>())
      return Ownership{ArrayReductionMode::BlockSharedNoOp, {}};
    if (auto forOp = value.getDefiningOp<scf::ForOp>())
      return classifyFor(forOp, value);
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Block *block = argument.getOwner();
      Operation *parent = block->getParentOp();
      if (auto computeRegion = dyn_cast<ComputeRegionOp>(parent)) {
        unsigned number = argument.getArgNumber();
        unsigned numLaunchArgs = computeRegion.getLaunchArgs().size();
        if (number < numLaunchArgs)
          return failure();
        return classify(computeRegion.getInputArgs()[number - numLaunchArgs]);
      }
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (argument.getArgNumber() == 0)
          return failure();
        return classifyFor(forOp, argument);
      }
      if (block == &parent->getRegion(0).front() &&
          isa<FunctionOpInterface>(parent)) {
        if (isa<gpu::GPUFuncOp>(parent))
          return failure();
        return Ownership{ArrayReductionMode::BlockSharedNoOp, {}};
      }
      return failure();
    }

    if (auto view = value.getDefiningOp<ViewLikeOpInterface>()) {
      if (view.getViewDest() != value)
        return failure();
      return classify(view.getViewSource());
    }
    if (auto partial = dyn_cast_or_null<PartialEntityAccessOpInterface>(
            value.getDefiningOp()))
      return classify(partial.getBaseEntity());

    if (auto select = value.getDefiningOp<arith::SelectOp>()) {
      if (std::optional<Value> selected = getConstantSelectedValue(select))
        return classify(*selected);
      return meet(select.getTrueValue(), select.getFalseValue());
    }
    if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
      unsigned resultNumber = cast<OpResult>(value).getResultNumber();
      if (std::optional<bool> condition =
              getConstantCondition(ifOp.getCondition())) {
        Region &selected =
            *condition ? ifOp.getThenRegion() : ifOp.getElseRegion();
        auto yield = cast<scf::YieldOp>(selected.front().getTerminator());
        return classify(yield.getOperand(resultNumber));
      }
      auto thenYield =
          cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto elseYield =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
      return meet(thenYield.getOperand(resultNumber),
                  elseYield.getOperand(resultNumber));
    }
    return failure();
  }

  FailureOr<Ownership> meet(Value lhs, Value rhs) {
    FailureOr<Ownership> lhsOwnership = classify(lhs);
    FailureOr<Ownership> rhsOwnership = classify(rhs);
    if (failed(lhsOwnership) || failed(rhsOwnership) ||
        lhsOwnership->mode != rhsOwnership->mode)
      return failure();
    lhsOwnership->privateLocals.insert(rhsOwnership->privateLocals.begin(),
                                       rhsOwnership->privateLocals.end());
    return lhsOwnership;
  }

  int64_t maxThreadPrivateStack;
  llvm::DenseMap<Value, Ownership> cache;
  llvm::DenseSet<Value> visiting;
};

static bool reductionHasBlockContext(ReductionAccumulateArrayOp op) {
  auto hasBlock = [](GPUParallelDimsAttr parDims) {
    return parDims && parDims.hasAnyBlockLevel();
  };
  if (hasBlock(op.getParDimsAttr()))
    return true;
  for (scf::ParallelOp loop = op->getParentOfType<scf::ParallelOp>(); loop;
       loop = loop->getParentOfType<scf::ParallelOp>())
    if (hasBlock(acc::getParDimsAttr(loop)))
      return true;
  return false;
}

class ACCPrepareCGToGPUPass
    : public acc::impl::ACCPrepareCGToGPUBase<ACCPrepareCGToGPUPass> {
public:
  using ACCPrepareCGToGPUBase::ACCPrepareCGToGPUBase;

  void runOnOperation() override {
    PolicyAnalyzer analyzer(maxThreadPrivateStack);
    WalkResult result =
        getOperation()->walk([&](ReductionAccumulateArrayOp op) {
          if (auto type = dyn_cast<MemRefType>(op.getMemref().getType());
              type && type.getRank() != 1) {
            op.emitError("array reduction requires a rank-1 memref");
            return WalkResult::interrupt();
          }
          bool hasThreadDim = false;
          bool hasThreadX = false;
          bool hasThreadY = false;
          bool hasThreadZ = false;
          for (GPUParallelDimAttr dim : op.getParDims().getArray()) {
            hasThreadDim |= dim.isAnyThread();
            hasThreadX |= dim.isThreadX();
            hasThreadY |= dim.isThreadY();
            hasThreadZ |= dim.isThreadZ();
          }
          if (hasThreadDim && !reductionHasBlockContext(op)) {
            op.emitError("reduction: thread-only array reduction accumulate");
            return WalkResult::interrupt();
          }
          auto preparedMode = op->getAttrOfType<ArrayReductionModeAttr>(
              ArrayReductionModeAttrName);
          auto preparedOperator = op->getAttrOfType<ReductionOperatorAttr>(
              ArrayReductionOperatorAttrName);
          auto isValidTopology = [&](ArrayReductionMode mode) {
            return (mode == ArrayReductionMode::BlockSharedNoOp ||
                    hasThreadX) &&
                   (mode != ArrayReductionMode::WorkerRow ||
                    (hasThreadY && !hasThreadZ));
          };
          if (preparedMode || preparedOperator) {
            if (!preparedMode || !preparedOperator ||
                preparedOperator.getValue() != op.getReductionOperator() ||
                (!hasThreadDim && op.getParDims().hasAnyBlockLevel() &&
                 preparedMode.getValue() !=
                     ArrayReductionMode::BlockSharedNoOp) ||
                !isValidTopology(preparedMode.getValue())) {
              op.emitError("invalid precomputed array reduction policy");
              return WalkResult::interrupt();
            }
          }
          if (!preparedMode && !hasThreadDim &&
              op.getParDims().hasAnyBlockLevel()) {
            preparedMode = ArrayReductionModeAttr::get(
                op.getContext(), ArrayReductionMode::BlockSharedNoOp);
            preparedOperator = op.getReductionOperatorAttr();
            op->setAttr(ArrayReductionModeAttrName, preparedMode);
            op->setAttr(ArrayReductionOperatorAttrName, preparedOperator);
          }
          bool blockOnlyPolicy =
              preparedMode &&
              preparedMode.getValue() == ArrayReductionMode::BlockSharedNoOp &&
              !hasThreadDim && op.getParDims().hasAnyBlockLevel();
          FailureOr<Ownership> ownership = analyzer.classify(op.getMemref());
          if (preparedMode && !blockOnlyPolicy && succeeded(ownership) &&
              preparedMode.getValue() != ownership->mode) {
            op.emitError("invalid precomputed array reduction policy");
            return WalkResult::interrupt();
          }
          if (failed(ownership)) {
            if (preparedMode)
              return WalkResult::advance();
            op.emitError("mixed or unsupported array storage ownership");
            return WalkResult::interrupt();
          }
          ArrayReductionMode mode =
              preparedMode ? preparedMode.getValue() : ownership->mode;
          if (!isValidTopology(mode)) {
            op.emitError("array reduction policy is incompatible with parallel "
                         "dimensions");
            return WalkResult::interrupt();
          }

          op->setAttr(ArrayReductionModeAttrName,
                      ArrayReductionModeAttr::get(op.getContext(), mode));
          op->setAttr(ArrayReductionOperatorAttrName,
                      op.getReductionOperatorAttr());
          for (PrivateLocalOp privateLocal : ownership->privateLocals) {
            auto privateType =
                cast<PrivateType>(privateLocal.getPrivatized().getType());
            if (auto memrefType = dyn_cast<MemRefType>(privateType.getBaseTy());
                memrefType && memrefType.getRank() != 1) {
              op.emitError("array reduction private storage must be rank-1");
              return WalkResult::interrupt();
            }
            PrivateStorageKind kind = analyzer.getStorageKind(privateLocal);
            if (Attribute old =
                    privateLocal->getAttr(PrivateStorageKindAttrName)) {
              if (old != PrivateStorageKindAttr::get(op.getContext(), kind)) {
                op.emitError("incompatible private storage policies");
                return WalkResult::interrupt();
              }
            }
            privateLocal->setAttr(
                PrivateStorageKindAttrName,
                PrivateStorageKindAttr::get(op.getContext(), kind));
            if (Attribute old =
                    privateLocal->getAttr(ArrayReductionOperatorAttrName)) {
              if (old != op.getReductionOperatorAttr()) {
                op.emitError("incompatible array reduction operators");
                return WalkResult::interrupt();
              }
            }
            privateLocal->setAttr(ArrayReductionOperatorAttrName,
                                  op.getReductionOperatorAttr());
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
