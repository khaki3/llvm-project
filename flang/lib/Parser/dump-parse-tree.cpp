//===-- lib/Parser/dump-parse-tree.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Holds the explicit instantiation of DumpTree for a whole Program. Walking the
// complete parse tree instantiates the visitor templates in
// parse-tree-visitor.h for every node type, so doing it here keeps that cost
// out of each caller.

#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::parser {

template llvm::raw_ostream &DumpTree<Program>(llvm::raw_ostream &out,
    const Program &x, const AnalyzedObjectsAsFortran *asFortran);

} // namespace Fortran::parser
