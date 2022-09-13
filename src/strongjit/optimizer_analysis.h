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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_ANALYSIS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_ANALYSIS_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "cppitertools/zip.hpp"
#include "strongjit/base.h"
#include "strongjit/block.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimizer_util.h"
#include "tuple_util.h"
#include "utils/diffs.h"
#include "utils/inlined_bit_vector.h"
#include "utils/status_macros.h"

// This file provide a standard way of creating a forward data-flow static
// analysis and then a rewriting pass based on this analysis.
//
// Each specific analysis, must provide a subclass of analysis::Base that
// provide all the required behavior. Thic subclass can also contains runtime
// attribute that affect its behavior. Then the Manager is initialized with all
// those analyses.
//
// Afterward the expected use case is to call Manager::Analyze to effectively do
// the analysis and then if there is a result, call RewriteAnalyse once or
// several time to do some rewrite depending on the results of the analysis.
// Manager::Analyze will not mutate the function.

namespace deepmind::s6 {

namespace analysis {

////////////////////////////////////////////////////////////////////////////////
// Analysis Base

// This class is the template interface that all analyses must provide.
// A concrete analysis must derive from this class and redefine all methods.
// The methods are protected so if a concrete analysis fails to override one
// method, there will be a compile error inside the manager.
//
// The template parameters have an underscore, so they can be reexported with a
// "using".
template <typename PointInfo_,
          typename DiffVecTraits_ = DiffVecTraits<PointInfo_>>
class Base {
 public:
  // The name of the analysis. Used for display by analysis::Annotator.
  static constexpr absl::string_view kName = "";

  // Store the analysis information at a specific point.
  using PointInfo = PointInfo_;

  // The DiffVecTraits of PointInfo. Define how PointInfo behave in a DiffVec.
  // It must be a valid DiffVec trait.
  using DiffVecTraits = DiffVecTraits_;

  // The Diff object over PointInfo.
  using Diff = typename DiffVecTraits::Diff;

  // The DiffVec Cursor. Useful to iterate over instructions.
  using InstCursor = typename DiffVec<
      TaggedDiffable<const Instruction*, PointInfo, DiffVecTraits>>::Cursor;

  using TerminatorDiffs = absl::InlinedVector<absl::optional<Diff>, 2>;

  // In all following methods, Context will be an instance of analysis::Context
  // which inherits from analysis::Manager so both API can be used on the
  // argument.
  // In particular, all the API of Manager that is marked as "Analysis only"
  // must only be called from those methods. Not every methods in this list can
  // call any API member, read their respective documentation. Making this
  // compile-time safe would be possible, but very code heavy and annoying.

 protected:
  // This method is called to initialize a block with an initial PointInfo,
  // that can depend on the context and the block. This will only be called once
  // per block, the first time the analysis sees the block. Any further update
  // will proceed by mutation and will not call this function again.
  template <typename Context>
  absl::StatusOr<PointInfo> InitializeBlock(Context& ctxt, const Block& b);

  // Processes the function entry. This is called only once on the entry block,
  // right after it has been initialized with InitializeBlock. The starting
  // PointInfo can be mutated to a valid state that correspond to function
  // entry.
  template <typename Context>
  absl::Status ProcessFunctionEntry(Context& ctxt, const Block& entry,
                                    PointInfo& info);

  // This method is called each time the analysis enters a block. If a block
  // starting info is updated because a later block jumps back to this block,
  // then this function will be called again each time the block is processed.
  template <typename Context>
  absl::Status ProcessBlockEntry(Context& ctxt, const Block& b,
                                 PointInfo& info);

  // Processes a single instruction. This method cannot mutate the PointInfo
  // directly. Instead it must return a Diff that will be applied to get the
  // PointInfo of the next instruction.
  template <typename Context>
  absl::StatusOr<Diff> ProcessInstruction(Context& ctxt,
                                          const Instruction& inst,
                                          const PointInfo& info);

  // Processes a terminator instruction. Is is called after ProcessInstruction
  // on a terminator instruction. That means that when processing a
  // TerminatorInst, both methods are called on it. It has again a mutable
  // access to the terminal PointInfo.
  // It must return a Diff by successor. If no Diff is present, it means that
  // taking the corresponding successor is impossible.
  template <typename Context>
  absl::StatusOr<TerminatorDiffs> ProcessTerminator(Context& ctxt,
                                                    const TerminatorInst& inst,
                                                    PointInfo& info);

  // The remaining methods are all called once on each successor of the
  // terminator instruction. In particular they are never called on
  // successor-less terminators like return.

  template <typename Context>
  absl::Status ProcessSuccessor(Context& ctxt, const TerminatorInst& inst,
                                const Block& successor, PointInfo& src_info);

  // Transfers a value from a source block to a destination block. This method
  // is called when src is transferred to dest in the destination block. Either
  // as a block argument or simply living through.
  //
  // This must return true if dest_info has been mutated and false otherwise.
  // The Manager will only reprocess the successor if dest_info has been
  // mutated. This is important for the convergence of the data-flow algorithm.
  //
  // src_info can be mutated freely and it will be seen by further
  // TransferValue. It is called first block arguments in their
  // specified order, then for live values in increasing value number order.
  template <typename Context>
  absl::StatusOr<bool> TransferValue(Context& ctxt, const Value* src,
                                     PointInfo& src_info, const Value* dest,
                                     PointInfo& dest_info);

  // Same as TransferValue but for the implicit argument of an except
  // instruction. In this case that means that the implicit argument named index
  // is transferred to the dest value in the successor block (which will
  // normally be an except or finally handler block). The boolean return value
  // has the same meaning as TransferValue.
  template <typename Context>
  absl::StatusOr<bool> TransferExceptImplicitValue(Context& ctxt, int64_t index,
                                                   const Value* dest,
                                                   PointInfo& dest_info);
};

// Checks if a type is a valid analysis.
template <typename P>
constexpr bool IsAnalysis =
    std::is_base_of_v<Base<typename P::PointInfo, typename P::DiffVecTraits>,
                      P>;

////////////////////////////////////////////////////////////////////////////////
// Exported types
//
// This section defines types that are useful to manipulate the output of the
// analysis.

// This class contains the information stored for each block during and after of
// the analysis. It is just a nice syntactic sugar over a DiffVec of PointInfos.
// It adds an Instruction* tag to each diff to remember at which instruction
// each diff should be applied.
template <typename Analysis>
class BlockInfo {
 public:
  static_assert((IsAnalysis<Analysis>),
                "Analysis must inherit from analysis::Base");

  using PointInfo = typename Analysis::PointInfo;
  using DiffVecTraits = typename Analysis::DiffVecTraits;
  using Cursor = typename Analysis::InstCursor;

  // The DiffVec instantiation encapsulated by this BlockInfo.
  using vec_type =
      DiffVec<TaggedDiffable<const Instruction*, PointInfo, DiffVecTraits>>;

  // Create a new block_info starting from this PointInfo.
  explicit BlockInfo(PointInfo start_info)
      : vec_({nullptr, std::move(start_info)}) {}

  // Returns the point information at the start of the block.
  // Since the information of the block is stored as a sequence of diffs,
  // mutating the start_info will implicitly change all other information.
  // Generally it is best to do a Restart in conjonction with a mutation of
  // start_info. In particular all Cursors will be invalidated.
  const PointInfo& start_info() const { return vec_.front().elem; }
  PointInfo& start_info() { return vec_.front().elem; }

  // Returns the underlying DiffVec.
  const vec_type& vec() const { return vec_; }
  vec_type& vec() { return vec_; }

  // Shortcuts to access the Cursor of the DiffVec.
  // x.BeginCursor() is the same as x.vec().BeginCursor().
  Cursor BeginCursor() const { return vec_.BeginCursor(); }
  Cursor EndCursor() const { return vec_.EndCursor(); }

  // Restarts the computation of this block because the start_info has been
  // updated with new information.
  // Only the start_info is preserved, any following information is discarded.
  void Restart() { vec_.resize(1); }

 private:
  vec_type vec_;
};

// Information stored per block as a result of the analysis.
struct CommonBlockInfo {
  // Whether this block was ever reached. If this is false, no analysis will
  // have any information about this block.
  bool reached = false;

  // Whether an instruction terminated the block before the end. This is
  // generally because an DeoptimizeIfSafepoint is taken unconditionaly. In
  // which case this will constain a pointer to the last instruction, which will
  // generally be a DeoptimizeIfSafepointInst.
  const Instruction* last_skipped = nullptr;
};

// The result of a the analysis for a specific analysis. Map blocks to
// BlockInfo. Is a block was not reached (As per CommonBlockInfo::reached), then
// there will be any binding for that block in the map.
template <typename Analysis>
using AnalysisResult = absl::flat_hash_map<const Block*, BlockInfo<Analysis>>;

// The common information for all the blocks.
using CommonResult = absl::flat_hash_map<const Block*, CommonBlockInfo>;

////////////////////////////////////////////////////////////////////////////////
// Analysis Manager

// The analysis manager. The normal protocol of use is:
// - Building all the individual analyses.
// - Building a manager from those analyses.
// - Call Analyze to perform all the analyses synchronously.
// - Use the analysis result, for example with rewrite analysis.
template <typename... Analyses>
class Manager {
 public:
  static_assert((IsAnalysis<Analyses> && ...),
                "All Analyses must inherit from analysis::Base");

  // A tuple of Cursors to represent the current point of the analysis,
  // inside a block.
  using Cursors = std::tuple<typename Analyses::InstCursor...>;

  // A tuple of point info to represent the full information at any given point.
  using PointInfos = std::tuple<typename Analyses::PointInfo...>;

  // Builds a Manager from a compile time set of analyses.
  explicit Manager(const Function& f, const PyCodeObject* code_object,
                   const LiveInfo& live_info, Analyses... analyses)
      : analyses_(std::move(analyses)...),
        live_info_(live_info),
        function_(f),
        code_object_(code_object) {}

  // The result of the analysis, contain an Analysis result for all analyses
  // plus the common result.
  class Result {
   public:
    // Returns the result of the specified analysis.
    template <typename Analysis>
    const AnalysisResult<Analysis>& Get() const {
      return std::get<AnalysisResult<Analysis>>(results_);
    }

    // Returns the common block information in this result.
    const CommonResult& GetCommon() const { return common_; }

   private:
    friend Manager;
    // Returns a multable access to the result of the analysis.
    // Used by the Manager to update the contained analysis.
    template <typename Analysis>
    AnalysisResult<Analysis>& GetMut() {
      return std::get<AnalysisResult<Analysis>>(results_);
    }

    // A tuple of the analysis results.
    std::tuple<AnalysisResult<Analyses>...> results_;

    // The common information for each block.
    CommonResult common_;
  };

  // Performs the analyses specified.
  absl::StatusOr<Result> Analyze();

  const LiveInfo& live_info() { return live_info_; }
  const ValueNumbering& numbering() { return live_info_.numbering(); }
  const Function& function() { return function_; }
  const PyCodeObject* code_object() { return code_object_; }

  //////////////////////////////////////////////////////////////////////////////
  // Analysis API.
  //
  // All the following methods until `private:` should only be called by the
  // methods provided by the Analyses (deriving from Base). They should not be
  // called from any external code that manipulates the manager. Doing so is UB
  // and will most likely result in a check failure or a segfault.

  // Asserts that the rest of the block is unreachable.
  // Do not process the next instruction (the current instruction will still
  // be processed by all analyses).
  // Should only be called from the Base::ProcessInstruction.
  void ExitBlock() {
    S6_CHECK(current_block_);
    S6_CHECK(current_instruction_);
    CommonBlockInfo& info = result().common_.at(current_block_);
    S6_CHECK(!info.last_skipped);
    info.last_skipped = current_instruction_;
  }

  // Forces a successor to be reprocessed. Can be called anywhere but should
  // logically only be called from Base::ProcessTerminator or Base::Transfer*
  void ReprocessSuccessor(const Block& successor) {
    worklist_.PushIfNew(&successor);
  }

  // Transfers information to the specified successor in a given analysis.
  // Calls the Base::Transfer* methods to get analysis-specific behavior.
  // Calls ReprocessSuccessor if any of Base::Transfer* methods signals a
  // modification.
  template <typename Analysis>
  absl::Status TransferToSuccessorGen(const Block& successor);

  // Returns the list of live value at a block entry.
  absl::Span<const Value* const> LiveValues(const Block& block) {
    return live_info_.LiveValues(block);
  }

  // Gets the starting information of a block for another analysis.
  template <typename Analysis>
  const typename Analysis::PointInfo& GetBlockStartInfo(const Block& b) {
    return result().template Get<Analysis>().GetBlockInfo(b).start_info();
  }

  // Checks if a block was reached.
  bool WasReached(const Block& b) {
    return result().GetCommon().at(&b).reached;
  }

  // Gets the current point information from another analysis.
  // Must only be called from Base::ProcessInstruction otherwise it is UB.
  template <typename Analysis>
  const typename Analysis::PointInfo& GetCurrentPointInfo() {
    return **std::get<typename Analysis::InstCursor>(cursors());
  }

  // Gets the point information of the current phase during termination of the
  // block.
  //
  // If this is called by ProcessTerminator. This will be generic information
  // either before of after the other analysis ProcessTerminator depending on
  // Analysis order.
  //
  // If this is called by either a *ProcessSuccessor or a Transfer* method, it
  // will give successor specific information.
  //
  // If this is called from any other analysis method, it is UB.
  template <typename Analysis>
  const typename Analysis::PointInfo& GetTerminatingPointInfo() {
    return std::get<typename Analysis::PointInfo>(terminating_infos());
  }

 private:
  // Mutable version of GetBlockStartInfo
  template <typename Analysis>
  typename Analysis::PointInfo& GetBlockStartInfoMut(const Block& b) {
    return result().template GetMut<Analysis>().at(&b).start_info();
  }

  // Mutable version of GetTerminatingPointInfo.
  template <typename Analysis>
  typename Analysis::PointInfo& GetTerminatingPointInfoMut() {
    return std::get<typename Analysis::PointInfo>(terminating_infos());
  }

  // Initializes a block by calling Base::InitializeBlock on all analyses
  // and storing the results. The block must not have been initialized before.
  // Being initialized is equivalent to having the `reached` flag set in the
  // common information. In particular this function requires that this flag
  // is unset for the block parameter and sets it.
  absl::Status InitializeBlock(const Block& block);

  // The list of analysis.
  std::tuple<Analyses...> analyses_;

  const LiveInfo& live_info_;
  const Function& function_;
  const PyCodeObject* code_object_;

  // The worklist of the data-flow algorithm is here and not just a local
  // variable of Analyze because some method of the Analysis API need to access
  // it.
  Worklist<const Block*> worklist_;

  // The current Result. Only set when we are inside the Analyze function.
  Result* result_ = nullptr;
  Result& result() {
    S6_DCHECK(result_);
    return *result_;
  }

  const Block* current_block_ = nullptr;
  const Instruction* current_instruction_ = nullptr;

  // This field contains cursors when processing instructions and directly
  // PointInfo when processing terminators.
  absl::variant<std::monostate, Cursors, PointInfos> current_data_;
  Cursors& cursors() { return absl::get<Cursors>(current_data_); }
  PointInfos& terminating_infos() {
    return absl::get<PointInfos>(current_data_);
  }
};

template <typename... Analyses>
Manager(const Function& f, const PyCodeObject* code_object,
        const LiveInfo& live_info, Analyses... analyses)
    -> Manager<Analyses...>;

// The context is was is passed as first parameter to all methods of an
// analysis. It is just a wrapper around the Manager itself that knows at
// compile time which analysis we are currently processing.
template <typename Analysis, typename... Analyses>
class Context : public Manager<Analyses...> {
 public:
  // Same as TransferToSuccessorGen but already specialized to the current
  // analysis.
  const absl::Status TransferToSuccessor(const Block& successor) {
    return this->template TransferToSuccessorGen<Analysis>(successor);
  }
};

// Specializes a manager into a context for a specific analysis.
template <typename Analysis, typename... Analyses>
Context<Analysis, Analyses...>& Specialize(Analysis&,
                                           Manager<Analyses...>& mngr) {
  return static_cast<Context<Analysis, Analyses...>&>(mngr);
}

template <typename... Analyses>
absl::Status Manager<Analyses...>::InitializeBlock(const Block& block) {
  S6_CHECK(!WasReached(block));
  result().common_.at(&block).reached = true;
  return tuple::for_each_Status(
      [&](auto& analysis, auto& result) {
        S6_CHECK(!result.contains(&block));
        S6_ASSIGN_OR_RETURN(
            auto point_info,
            analysis.InitializeBlock(Specialize(analysis, *this), block));
        result.try_emplace(&block, std::move(point_info));
        return absl::OkStatus();
      },
      analyses_, result().results_);
}

template <typename... Analyses>
template <typename Analysis>
absl::Status Manager<Analyses...>::TransferToSuccessorGen(
    const Block& successor) {
  using PointInfo = typename Analysis::PointInfo;

  // This function can only be calles while processing a terminator, so we
  // can assume that we can get the terminating PointInfo,
  Analysis& analysis = std::get<Analysis>(analyses_);
  PointInfo& info = GetTerminatingPointInfoMut<Analysis>();

  // And we can also assume that the current instruction is a terminator.
  S6_CHECK(current_instruction_);
  const TerminatorInst& terminator =
      cast<TerminatorInst>(*current_instruction_);
  int num_implicit = terminator.num_implicit_arguments();

  // The flag will be set to true if any mutation is applied to information of
  // the successor and thus the successor need to be processes again.
  bool changed = false;

  // If this is the first time the block is entered, it needs to be processed
  // anyway.
  if (!WasReached(successor)) {
    S6_RETURN_IF_ERROR(InitializeBlock(successor));
    changed = true;
  }

  // Now that we are sure that the successor is initialized, we can access its
  // start information.
  PointInfo& succ_info = GetBlockStartInfoMut<Analysis>(successor);

  // First, if the terminator is an except, we propagate the implicit
  // value information.
  if (isa<ExceptInst>(terminator)) {
    for (int64_t i = 0; i < num_implicit; ++i) {
      S6_ASSIGN_OR_RETURN(bool new_changed,
                          analysis.TransferExceptImplicitValue(
                              Specialize(analysis, *this), i,
                              successor.block_arguments()[i], succ_info));
      changed = changed || new_changed;
    }
  } else {
    S6_CHECK_EQ(num_implicit, 0)
        << "Non ExceptInst implicits not supported yet";
  }

  // The we propagate the information of the live values.
  for (const Value* live : LiveValues(successor)) {
    S6_ASSIGN_OR_RETURN(bool new_changed,
                        analysis.TransferValue(Specialize(analysis, *this),
                                               live, info, live, succ_info));
    changed = changed || new_changed;
  }

  // And lastly, we can propagate the information of the block arguments.
  for (auto [source, destination] :
       iter::zip(terminator.successor_arguments(&successor),
                 successor.block_arguments().subspan(num_implicit))) {
    S6_ASSIGN_OR_RETURN(
        bool new_changed,
        analysis.TransferValue(Specialize(analysis, *this), source, info,
                               destination, succ_info));
    changed = changed || new_changed;
  }
  // If succ_info was indeed mutated, we need to reprocess `successor`.
  if (changed) ReprocessSuccessor(successor);
  return absl::OkStatus();
}

template <typename... Analyses>
auto Manager<Analyses...>::Analyze() -> absl::StatusOr<Result> {
  Result result;
  result_ = &result;

  // We want to have the common block information for all blocks.
  for (const Block& block : function_) result.common_.try_emplace(&block);

  // First we do the special processing of the function entry.
  current_block_ = &function_.entry();

  S6_RETURN_IF_ERROR(InitializeBlock(function_.entry()));

  S6_RETURN_IF_ERROR(tuple::for_each_Status(
      [&](auto& analysis, auto& result) {
        return analysis.ProcessFunctionEntry(
            Specialize(analysis, *this), function_.entry(),
            result.at(&function_.entry()).start_info());
      },
      analyses_, result.results_));

  // We start by processing the function entry.
  worklist_.Push(&function_.entry());

  // This is the main loop of the forward-data-flow algorithm. See the wikipedia
  // article: https://en.wikipedia.org/wiki/Data-flow_analysis. In particular
  // the section "The work list approach". This algorithm is a slight
  // improvement over that section, because blocks are reprocessed only when one
  // of their predessors adds new information to their starting conditions.
  // TODO: Think about the processing order if the performance is
  // getting problematic
  while (!worklist_.empty()) {
    const Block& b = *worklist_.Pop();
    current_block_ = &b;

    // Initialize the block if it was not reached.
    if (!WasReached(b)) S6_RETURN_IF_ERROR(InitializeBlock(b));

    CommonBlockInfo& common_info = result.common_.at(&b);
    // Resets the early exit point of the block which is now invalid as the
    // block has been updated with new information.
    common_info.last_skipped = nullptr;

    // Processes the block entry.
    S6_RETURN_IF_ERROR(tuple::for_each_Status(
        [&](auto& analysis, auto& result) {
          result.at(&b).Restart();
          return analysis.ProcessBlockEntry(Specialize(analysis, *this), b,
                                            result.at(&b).start_info());
        },
        analyses_, result.results_));

    // Set the current point in current_data_ as a tuple of the `begin`
    // iterators.
    S6_CHECK(absl::holds_alternative<std::monostate>(current_data_));
    current_data_ = tuple::transform(
        [&](auto& result) { return result.at(&b).BeginCursor(); },
        result.results_);

    // Iterates of all the instruction of the block to propagate the
    // information from the start to the end of the block.
    for (const Instruction& inst : b) {
      current_instruction_ = &inst;

      S6_ASSIGN_OR_RETURN(
          auto diffs, tuple::transform_StatusOr(
                          [&](auto& analysis, const auto& cursor) {
                            return analysis.ProcessInstruction(
                                Specialize(analysis, *this), inst, **cursor);
                          },
                          analyses_, cursors()));

      // Apply the diffs and move the iterators after that diff.
      tuple::for_each(
          [&](auto&& diff, auto& result, auto& cursor) {
            if (diff.empty()) return;
            // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
            result.at(&b).vec().push_back_maintain({&inst, std::move(diff)},
                                                   cursor);
            cursor.StepForward();
          },
          std::move(diffs), result.results_, cursors());

      // If one of the `ProcessInstruction` called `ExitBlock`, we do an
      // early exit of the block.
      if (common_info.last_skipped) {
        S6_CHECK_EQ(common_info.last_skipped, &inst);
        break;
      }

      // If we are processing a normal non-terminator instruction, move to the
      // next instruction.
      if (!isa<TerminatorInst>(inst)) continue;

      // If this was unreachable, do nothing else in this block and exit.
      // It is fine to reach an unreachable instruction, it just means that the
      // static analyses were unable to prove that this is indeed unreachable
      // which is perfectly fine.
      if (isa<UnreachableInst>(inst)) break;

      const TerminatorInst& terminator = cast<TerminatorInst>(inst);
      const int64_t num_successor = terminator.successor_size();

      // We take the final PointInfos out of the iterators since they are
      // immutable in the iterators.
      current_data_ = tuple::transform(
          [](auto&& cursor) {
            return **std::move(cursor);  // NOLINT
          },
          std::move(cursors()));

      // Process each terminator. They each return an array of diffs, that need
      // to be applied to the respective PointInfo to get the successor pre-jump
      // info.
      S6_ASSIGN_OR_RETURN(auto succ_diffs,
                          tuple::transform_StatusOr(
                              [&](auto& analysis, auto& result, auto& info) {
                                return analysis.ProcessTerminator(
                                    Specialize(analysis, *this), terminator,
                                    info);
                              },
                              analyses_, result.results_, terminating_infos()));

      if (num_successor == 0) break;

      // Compute the list of valid successor. A successor is invalid if at least
      // one analysis return nullopt instead of a Diff for that specific
      // successor and it means that this analysis thinks the successor is
      // unreachable.
      InlinedBitVector<2> valid_succ(num_successor);
      valid_succ.SetBits(0, num_successor);
      tuple::for_each(
          [&](const auto& diff_array) {
            S6_CHECK_EQ(diff_array.size(), num_successor);
            for (int64_t i = 0; i < num_successor; ++i) {
              if (!diff_array[i].has_value()) valid_succ.clear_bit(i);
            }
          },
          succ_diffs);

      // If there are no valid successor there is no need to continue.
      if (valid_succ.count() == 0) break;

      // I need the last_valid successor to avoid a extra-copy of the PointInfo
      int64_t last_valid_successor;
      for (int64_t i = 0; i < num_successor; ++i) {
        if (valid_succ[i]) last_valid_successor = i;
      }

      // Save the generic terminator information in infos. I will then place
      // successor specific information in terminating_infos().
      PointInfos infos = std::move(terminating_infos());
      for (int64_t succ_index = 0; succ_index < num_successor; ++succ_index) {
        if (!valid_succ[succ_index]) continue;

        const Block* successor = terminator.successors()[succ_index];

        // I want a copy onf infos in terminating_infos() but it this is the
        // last valid successor, I can do a move instead.
        if (succ_index == last_valid_successor) {
          terminating_infos() = std::move(infos);
        } else {
          terminating_infos() = infos;
        }

        // The I apply the diff on terminating_infos() to get successor specific
        // information.
        tuple::for_each(
            [&](auto& analysis, auto& info, const auto& diffs) {
              using Analysis = std::remove_reference_t<decltype(analysis)>;
              Analysis::DiffVecTraits::Apply(info, *diffs[succ_index]);
            },
            analyses_, terminating_infos(), succ_diffs);

        // And finally I can run ProcessSuccessor to process each successor with
        // the successor specific information.
        S6_RETURN_IF_ERROR(tuple::for_each_Status(
            [&](auto& analysis, auto& result, auto& info) {
              return analysis.ProcessSuccessor(Specialize(analysis, *this),
                                               terminator, *successor, info);
            },
            analyses_, result.results_, terminating_infos()));
      }

      break;
    }  // End of instruction loop.

    // Reset all block-specific state in the Manager.
    current_instruction_ = nullptr;
    current_block_ = nullptr;
    current_data_ = std::monostate{};
  }  // End of block worklist loop

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Analysis Formatter

// An AnnotationFormatter for analyses results.
//
// REQUIREMENT: Analysis::PointInfo must provide a ToString free function that
// takes itself and a ValueNumbering. This method will be dicovered by ADL.
template <typename Analysis>
class Annotator final : public AnnotationFormatter {
 public:
  std::string FormatAnnotation(const Value& v, FormatContext* ctx) const final {
    if (const Block* b = dyn_cast<Block>(&v)) {
      if (!result_.contains(b)) return "";
      return absl::StrCat(
          Analysis::kName, ": ",
          ToString(result_.at(b).start_info(), ctx->value_numbering()));
    }

    if (const Instruction* inst = dyn_cast<Instruction>(&v)) {
      const Block& b = *inst->parent();
      if (!result_.contains(&b)) return "";
      const BlockInfo<Analysis>& info = result_.at(&b);
      auto cur = info.vec().Find(
          [inst](const auto& tagged_elem) { return tagged_elem.tag == inst; });
      if (cur.IsEnd()) return "";
      return absl::StrCat(Analysis::kName, ": ",
                          ToString(**cur, ctx->value_numbering()));
    }

    return "";
  }
  explicit Annotator(const AnalysisResult<Analysis>& result)
      : result_(result) {}

 private:
  const AnalysisResult<Analysis>& result_;
};

template <class Analysis>
Annotator(const AnalysisResult<Analysis>&) -> Annotator<Analysis>;

////////////////////////////////////////////////////////////////////////////////
// Rewriting based on analysis result.

// This class defines the interface of analysis-based rewriter.
// A concrete rewriter must derive from this class and redefine all methods.
// The methods are protected so if a concrete analysis fails to override one
// method, there will be a compile error inside the manager.
//
// The Analyses parameter pack is the list of analyses on which the rewriter
// depends.
template <typename... Analyses>
class RewriterBase {
 public:
  using Results = std::tuple<const AnalysisResult<Analyses>&...>;
  using BlockInfos = std::tuple<const BlockInfo<Analyses>&...>;
  using PointInfos = std::tuple<const typename Analyses::PointInfo&...>;

  // Diffs are pointers and not references here, because they are optional.
  using Diffs = std::tuple<const typename Analyses::Diff*...>;

  // Extract the result on which this rewriter depends from a generic result.
  // `Result` should of type `Manager::Result`.
  template <typename Result>
  static Results GetRelevantResults(const Result& result) {
    return Results(result.template Get<Analyses>()...);
  }

  const LiveInfo& live_info() const {
    S6_DCHECK(live_info_);
    return *live_info_;
  }

 protected:
  // Each of the rewrite method should only mutate the function locally.
  // For block arguments, and live values, this means only inserting instruction
  // at the start of the block.
  // For instructions, this means only inserting instruction just before and
  // after the target instruction and/or deleting only that instruction.

  void RewriteBlockArgument(Rewriter& rewriter, PointInfos infos,
                            BlockArgument* arg);
  void RewriteLiveValue(Rewriter& rewriter, PointInfos infos, Block& block,
                        Value* live);

  // Rewrites an instruction. The `infos` arguments is the PointInfo from before
  // the instruction. The `diffs` arguments are the diffs generated by that
  // instruction. Each diff is a pointer that may be null if no diff was
  // generated by that instruction.
  void RewriteInstruction(Rewriter& rewriter, PointInfos infos, Diffs diffs,
                          Instruction& inst);

 private:
  template <typename AnalysisRewriter, typename Result>
  friend void RewriteAnalysis(Rewriter& rewriter, const LiveInfo& live_info,
                              AnalysisRewriter arewriter, const Result& result);
  const LiveInfo* live_info_ = nullptr;
};

// Applies a rewriter based on RewriterBase to a function.
template <typename AnalysisRewriter, typename Result>
void RewriteAnalysis(Rewriter& rewriter, const LiveInfo& live_info,
                     AnalysisRewriter arewriter, const Result& result) {
  Function& f = rewriter.function();
  CommonResult common_result = result.GetCommon();

  arewriter.live_info_ = &live_info;
  using Results = typename AnalysisRewriter::Results;
  using BlockInfos = typename AnalysisRewriter::BlockInfos;
  using PointInfos = typename AnalysisRewriter::PointInfos;
  using Diffs = typename AnalysisRewriter::Diffs;

  Results results = AnalysisRewriter::GetRelevantResults(result);

  for (Block& b : f) {
    const CommonBlockInfo& common_info = result.GetCommon().at(&b);
    if (!common_info.reached) continue;
    BlockInfos block_infos = tuple::transform(
        [&](const auto& result) -> const auto& { return result.at(&b); },
        results);

    PointInfos start_infos = tuple::transform(
        [&](const auto& block_info) -> const auto& {
          return block_info.start_info();
        },
        block_infos);

    // Initialiaze the block walking iterator here, so that instruction
    // added by RewriteBlockArgument or RewriteLiveValue are not iterated over.
    auto it = b.begin();

    for (BlockArgument* arg : b.block_arguments()) {
      arewriter.RewriteBlockArgument(rewriter, start_infos, arg);
    }
    for (const Value* live : live_info.LiveValues(b)) {
      // The const cast is fine because we own a non-const reference to the
      // function.
      arewriter.RewriteLiveValue(rewriter, start_infos, b,
                                 const_cast<Value*>(live));
    }

    auto cursors = tuple::transform(
        [&](const auto& block_info) { return block_info.BeginCursor(); },
        block_infos);

    // TODO: Maybe uniformize this with the pattern rewriter.
    // The pattern rewriter uses a Cursor whereas this loop handle the iterator
    // manually.
    while (it != b.end()) {
      // We increment the iterator here, so that the rest of the code,
      // can delete `inst`, or create instruction around it and it won't
      // affect the iterator `it`. Postfix increment is by design here.
      Instruction& inst = *it++;

      PointInfos infos = tuple::transform(
          [&](const auto& cursor) -> decltype(auto) { return **cursor; },
          cursors);

      Diffs diffs = tuple::transform(
          [&](const auto& cursor) -> const decltype(cursor.NextDiff()->diff)* {
            auto ndiff = cursor.NextDiff();
            if (!ndiff) return nullptr;
            if (ndiff->tag != &inst) return nullptr;
            return &ndiff->diff;
          },
          cursors);

      arewriter.RewriteInstruction(rewriter, infos, diffs, inst);

      // Bumps all iterators for which we have a diff.
      tuple::for_each(
          [&](auto& cursor, auto diff) {
            if (diff) cursor.StepForward();
          },
          cursors, diffs);

      // If this is the last instruction of the block to run, stop here.
      if (common_info.last_skipped == &inst) break;
    }  // End of instruction loop
  }    // End of block loop
}

}  // namespace analysis

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_ANALYSIS_H_
