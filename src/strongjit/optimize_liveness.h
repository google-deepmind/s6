// Copyright 2021 The s6 Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_LIVENESS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_LIVENESS_H_

#include <string>

#include "absl/strings/str_join.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"
#include "strongjit/value_map.h"
#include "utils/inlined_bit_vector.h"

namespace deepmind::s6 {

// The goal of the liveness analysis is to compute which value may be used by
// a block and its successors in addition to its block arguments.
// After the analysis there is a guarantee that no values other than the
// block arguments and the live values outputed by this analysis are usable by
// the following code.

// Compute the Liveness information and provide it. Also provides and maintains
// a value numbering.
class LiveInfo {
 public:
  // Performs the live info analysis on a function f.
  explicit LiveInfo(const Function& f);

  // Checks if a variable is live at at a certain block entry.
  bool IsLiveAtEntry(const Block& block, const Value* target) const {
    S6_CHECK(infos_.contains(&block));
    return infos_.at(&block).bits[numbering_.at(target)];
  }

  // Returns a string version of the live values at block entry as a space
  // separated list.
  std::string ToString(const Block& block) const {
    auto& list = infos_.at(&block).list;
    if (list.empty()) return "";
    return absl::StrJoin(list, " ", [&](std::string* out, const Value* val) {
      absl::StrAppend(out, "%", numbering_.at(val));
    });
  }

  // Returns the list of live values at block entry.
  absl::Span<const Value* const> LiveValues(const Block& block) const {
    return infos_.at(&block).list;
  }

  const ValueNumbering& numbering() const { return numbering_; }

 private:
  // Representation of the live value at a block entry. Values are addressed by
  // number (from numbering_) in this bitvector. A set bit indicates that the
  // value is live at the block entry.
  using BlockInfoBits = InlinedBitVector<128>;

  // We use two representations for live info. The bits version is used during
  // computation and for fast access while the vector version is used for
  // iterating on live values.
  struct BlockInfo {
    BlockInfoBits bits;
    absl::InlinedVector<const Value*, 3> list;

    // Fill the `list` member from the `bits` member. `list` should be empty.
    void FillListFromBits(const std::vector<const Value*>& reverse_numbering) {
      S6_CHECK(list.empty());
      list.reserve(bits.count());
      for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) list.push_back(reverse_numbering[i]);
      }
    }
  };

  void FillListFromBits() {
    auto rvn = ReverseValueNumbering(numbering_);
    for (auto& [block, info] : infos_) {
      info.FillListFromBits(rvn);
    }
  }

  absl::flat_hash_map<const Block*, BlockInfo> infos_;
  ValueNumbering numbering_;
};

// Annotates a function with the live values at block entry.
// For example : &4: [ %5, %6 ]          // live values: %1 %3
class LiveInfoAnnotator final : public AnnotationFormatter {
 public:
  std::string FormatAnnotation(const Value& v, FormatContext* ctx) const final {
    if (auto b = dyn_cast<Block>(&v)) {
      return absl::StrCat("live values: ", li_.ToString(*b));
    }
    return "";
  }
  explicit LiveInfoAnnotator(const LiveInfo& li) : li_(li) {}

 private:
  const LiveInfo& li_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_LIVENESS_H_
