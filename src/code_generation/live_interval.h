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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_LIVE_INTERVAL_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_LIVE_INTERVAL_H_

#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "core_util.h"
#include "strongjit/formatter.h"
#include "strongjit/value.h"

namespace deepmind::s6 {

// Register allocation is built around the concept of a "live-interval". This is
// a range for which a single SSA value is live in a particular location -
// register or stack. A single SSA value may have multiple live-intervals as it
// is moved between locations but always starts with one.
//
// A live-interval is defined over "slot indices", which is a numbering over
// all instructions in the function. The numbering is arbitrary as long as it
// is a total order in which defs precede uses. Different numberings can affect
// the efficacy of register allocation but not its correctness.
//
// For example:
//   Slot  Instruction                  Live interval
//   0     %0 = frame_variable consts   %0 = [0, 2)
//   1     %1 = add %0, %0              %1 = [1, 2)
//   2     jmp &3 [%1, %0]
//   3     &3: [%4,                     %4 = [3, 5)  Note each block argument
//   4          %5]                     %5 = [4, 5)  gets its own slot.
//   5    %6 = add %4, %5               %6 = [5, 6)
//   6    return %6

// Forward-declare.
class LiveInterval;

// A location requirement constrains the possible choices of the register
// allocator. Simple requirements mandate that a certain register is used, or
// that any location may be used as long as it is a register.
class LocationRequirement {
 public:
  // Requires a specific location.
  static LocationRequirement InLocation(Location loc) {
    return LocationRequirement(loc, kLocation);
  }

  // Must be in a frame slot.
  static LocationRequirement InFrameSlot() {
    return LocationRequirement({}, kAnyFrameSlot);
  }

  // Has no requirement.
  static LocationRequirement Anywhere() {
    return LocationRequirement({}, kAnywhere);
  }

  // Requires a register, but it could be any register.
  static LocationRequirement InRegister() {
    return LocationRequirement({}, kAnyRegister);
  }

  // Returns true if the location may be anywhere.
  bool IsAnywhere() const { return kind_ == kAnywhere; }

  // Returns true if the requirement has a specific location.
  bool IsLocation() const { return kind_ == kLocation; }

  // Returns true if the requirement just has to be in some register (or
  // immediate).
  bool IsInRegister() const { return kind_ == kAnyRegister; }

  // Returns true if the location must be in a frame slot.
  bool IsInFrameSlot() const { return kind_ == kAnyFrameSlot; }

  const Location& location() const {
    S6_CHECK(IsLocation());
    return *location_;
  }

  std::string ToString() const {
    switch (kind_) {
      case kAnywhere:
        return "anywhere";
      case kAnyFrameSlot:
        return "any-frame-slot";
      case kLocation:
        return location().ToString();
      case kAnyRegister:
        return "any-register";
    }
    S6_UNREACHABLE();
  }

  // Returns true if `loc` adheres to this requirement.
  // Note that IsImmediate() is special-cased and will return true for
  // IsInRegister() and IsInFrameSlot().
  // TODO: This is legacy contract behavior, make that more explicit in
  // the function name.
  bool AdheresToRequirement(Location loc) const {
    switch (kind_) {
      case kAnywhere:
        return true;
      case kAnyFrameSlot:
        return loc.IsFrameSlot() || loc.IsImmediate();
      case kLocation:
        return location() == loc;
      case kAnyRegister:
        return loc.IsInRegister() || loc.IsImmediate();
    }
    S6_UNREACHABLE();
  }

  LocationRequirement() : kind_(kAnywhere) {}
  explicit LocationRequirement(Location loc)
      : LocationRequirement(loc, kLocation) {}

 private:
  enum Kind {
    kAnywhere,
    kAnyFrameSlot,
    kAnyRegister,
    kLocation,
  };

  LocationRequirement(absl::optional<Location> loc, Kind kind)
      : location_(loc), kind_(kind) {}

  absl::optional<Location> location_;
  Kind kind_;
};

// A live range is the unit of register allocation. It represents a [start, end)
// half-open interval in terms of slot indexes in which a Location is used.
// The interval is live on exit from (defined at) `start` and on entry to (last
// used at) `end`.
//
// A LiveRange is a view into a LiveInterval, so should always be passed by
// value. It is never invalidated by mutations to LiveInterval (unless the
// LiveRange is explicitly destroyed).
class LiveRange {
 public:
  explicit LiveRange(LiveInterval* interval, int64_t index)
      : interval_(interval), index_(index) {}

  // Returns the slot at which this interval is defined.
  int64_t startpoint() const;

  // Returns the slot at which this interval is last used.
  int64_t endpoint() const;

  // Returns the location associated with this interval.
  const Location& location() const;
  void set_location(const Location& loc);

  // Returns the hint associated with this range, if one exists.
  const LocationRequirement& hint() const;
  // Overwrites the hint for this range.
  void set_hint(const LocationRequirement& loc);

  const LiveInterval& interval() const { return *interval_; }

  std::string ToString() const {
    return absl::StrCat("[", startpoint(), ", ", endpoint(), ")");
  }

  // Returns the next LiveRange in the parent interval, if one exists.
  absl::optional<LiveRange> Next() const;

  // Splits this live range at `split_point`, creating two live ranges:
  //  [..., S), [S, ...]. This live range becomes the first range of the split.
  //
  // Both this and the new range are returned. The newly split live range does
  // not have an assigned location.
  std::pair<LiveRange, LiveRange> Split(int64_t split_point);

  bool operator<(const LiveRange& other) const {
    return std::make_tuple(startpoint(), endpoint()) <
           std::make_tuple(other.startpoint(), other.endpoint());
  }
  bool operator>(const LiveRange& other) const {
    return std::make_tuple(startpoint(), endpoint()) >
           std::make_tuple(other.startpoint(), other.endpoint());
  }
  bool operator==(const LiveRange& other) const {
    return std::make_tuple(startpoint(), endpoint()) ==
           std::make_tuple(other.startpoint(), other.endpoint());
  }

  friend std::ostream& operator<<(std::ostream& out, const LiveRange& r) {
    out << r.ToString();
    return out;
  }

 private:
  LiveInterval* interval_;
  int64_t index_;
};

// A LiveInterval represents the location of a single SSA value over its
// lifetime as a sequence of LiveRanges.
//
// A simple interval contains a single LiveRange, but more complex intervals
// may contain multiple ranges to encode live-range holes (where the value is
// not live) or to represent splits where the location containing a value
// changes (for example a region where the value is spilled).
class LiveInterval {
 public:
  // Returns the SSA value that this interval contains.
  const Value* value() const { return value_; }

  std::string ToString() const {
    std::string s =
        absl::StrJoin(ranges_, ", ", [](std::string* s, const auto& entry) {
          absl::StrAppend(s, "[", entry.startpoint, ", ", entry.endpoint,
                          ") @ ", entry.location.ToString());
        });
    return absl::StrCat(FormatOrDie(*value()), ": ", s);
  }

  // The startpoint of the interval, which is the minimum startpoint of all
  // constituent ranges.
  int64_t startpoint() const { return ranges_.front().startpoint; }

  // The endpoint of the interval, which is the maximum endpoint of all
  // constituent ranges.
  int64_t endpoint() const { return ranges_.back().endpoint; }

  // Returns the Location that this interval has on entry to `slot`.
  //
  // Because of use points, an interval may have multiple locations at a
  // particular slot. Consider a call:
  //   %1 = ...       // def(rdi)
  //   call_native %1 // use(rdi) copies(rdi -> stack[0])
  //   ... = use %1   // use(stack[0])
  //
  // As %1 must be used in rdi for the call but also preserved for the later
  // use, it is alive in two locations. In this case the live range information
  // will always contain the spilled location (the stack location). Copies
  // will have been inserted to ensure rdi is correct.
  //
  // When such a situation occurs, this function returns the *register*, not
  // the stack location.
  Location GetLocationOnEntryTo(int64_t slot) const {
    auto use_it = absl::c_lower_bound(
        use_points_, slot,
        [](const auto& p, int64_t s) { return p.first < s; });
    if (use_it != use_points_.end() && use_it->first == slot &&
        use_it->second.IsLocation()) {
      return use_it->second.location();
    }
    return GetRangeLocationOnEntryTo(slot);
  }

  // See GetLocationOnEntryTo(); this is identical but where an interval may
  // have two locations this returns the *stack location*, not the register.
  //
  // This is only expected to be useful for use-point copy insertion code that
  // requires both the stack and register locations.
  Location GetRangeLocationOnEntryTo(int64_t slot) const {
    // lower_bound gets us the range *after* slot ends.
    auto it = absl::c_lower_bound(
        ranges_, LiveRangeImpl{slot, -1, Location::Undefined(),
                               LocationRequirement::Anywhere()});
    S6_CHECK(it != ranges_.begin())
        << "bad query slot " << slot << " " << ToString();
    --it;

    return it->location;
  }

  LiveRange GetLiveRangeAt(int64_t slot) const {
    // upper_bound gets us the range *after* slot ends.
    auto it = absl::c_upper_bound(
        ranges_,
        LiveRangeImpl{slot, /*endpoint=*/INT64_MAX, Location::Undefined(),
                      LocationRequirement::Anywhere()});
    S6_CHECK(it != ranges_.begin())
        << "bad query slot " << slot << " " << ToString();
    --it;

    return LiveRange(const_cast<LiveInterval*>(this),
                     std::distance(ranges_.begin(), it));
  }

  // Returns the Location that this interval begins with.
  const Location& GetInitialLocation() const {
    return ranges_.begin()->location;
  }

  // Sets the location to `loc` for the entire interval. This erases all
  // sub-ranges and replaces them with a single range.
  void SetLocationForAllRanges(Location loc) {
    int64_t start = ranges_.begin()->startpoint;
    int64_t end = ranges_.rbegin()->endpoint;
    ranges_.clear();
    ranges_.push_back({start, end, loc, LocationRequirement::Anywhere()});
  }

  // Adds a new live range with an empty location.
  //
  // REQUIRES: The new live range does not overlap with any existing ranges.
  void AddRange(int64_t start, int64_t end,
                LocationRequirement hint = LocationRequirement::Anywhere()) {
    LiveRangeImpl range = {start, end, Location::Undefined(), hint};
    ranges_.insert(absl::c_upper_bound(ranges_, range), range);
  }

  int64_t num_live_ranges() const { return ranges_.size(); }
  LiveRange live_range(int64_t i) const {
    return LiveRange(const_cast<LiveInterval*>(this), i);
  }

  // Adds a new use point with the given location requirement.
  //
  // REQUIRES: This must be called in ascending order of use_point.
  void AddUsePoint(int64_t use_point, LocationRequirement loc) {
    use_points_.emplace_back(use_point, loc);
  }

  // Returns the use points, ordered.
  const absl::InlinedVector<std::pair<int64_t, LocationRequirement>, 4>&
  use_points() const {
    return use_points_;
  }

  // A use point, where an SSA value is used. A use point may not be in the SSA
  // value's live interval; it could be in a chained live interval.
  struct UsePoint {
    // The interval this use point is located within.
    const LiveInterval* interval;
    // The use point, as a slot index.
    int64_t point;
    // The location hint at this use point.
    LocationRequirement location_hint;
  };

  // Returns the next use point after `point`, or nullopt if there are no more
  // use points.
  absl::optional<UsePoint> NextUsePoint(int64_t point) const {
    auto it = absl::c_upper_bound(
        use_points_, point,
        [](int64_t a, const auto& b) { return a < b.first; });
    if (it != use_points_.end() && point < endpoint())
      return UsePoint{
          .interval = this, .point = it->first, .location_hint = it->second};
    if (outgoing_chain_) {
      return outgoing_chain_->NextUsePoint(point);
    }
    return absl::nullopt;
  }

  // Returns the next location hint after `point`. Conceptually this iterates
  // through use points until a use point has a required location.
  Location GetNextUsedLocation(int64_t point) const {
    auto it = absl::c_upper_bound(
        use_points_, point,
        [](int64_t a, const auto& b) { return a < b.first; });
    for (; it != use_points_.end(); ++it) {
      if (it->second.IsLocation()) return it->second.location();
    }
    if (outgoing_chain_) {
      return outgoing_chain_->GetNextUsedLocation(point);
    }
    return Location::Undefined();
  }

  // A live interval may be chained with another interval. This occurs if an
  // interval dies and immediately feeds a block argument. An interval with
  // an incoming chain will use the chain's last live range's location as an
  // initial hint to avoid copies.
  LiveInterval* incoming_chain() const { return incoming_chain_; }
  LiveInterval* outgoing_chain() const { return outgoing_chain_; }

  // Sets this chain's outgoing_chain to `chain` and sets `chain`'s
  // incoming_chain to this.
  //
  // For now, one can only chain with another LiveInterval if it starts
  // later than the current LiveInterval. This restriction can only be
  // lifted if a smarter way of preventing LiveInterval loops is found.
  // If the constraint is not respected ChainTo will silently do nothing.
  void ChainTo(LiveInterval* chain) {
    if (startpoint() >= chain->startpoint()) return;
    outgoing_chain_ = chain;
    chain->incoming_chain_ = this;
  }

  // Creates a live interval for `v` with no live ranges.
  explicit LiveInterval(const Value* v = nullptr) : value_(v) {}

 private:
  // LiveRange can manipulate the ranges_ list.
  friend class LiveRange;

  // The implementation of a live-range. The LiveRange class is the public
  // front to this that holds an index into the ranges_ list for iterator
  // stability.
  struct LiveRangeImpl {
    int64_t startpoint;
    int64_t endpoint;
    Location location;
    LocationRequirement hint;

    bool operator<(const LiveRangeImpl& other) const {
      return std::make_tuple(startpoint, endpoint) <
             std::make_tuple(other.startpoint, other.endpoint);
    }
  };

  // The implementation of LiveRange::Split. This invalidates all live ranges
  // after the range being split.
  std::pair<LiveRange, LiveRange> SplitRange(int64_t index,
                                             int64_t split_point) {
    int64_t e = ranges_[index].endpoint;
    LiveRangeImpl new_range = {split_point, e, Location(),
                               LocationRequirement::Anywhere()};
    S6_CHECK_GT(e, split_point) << "Zero-sized splits disallowed, splitting "
                                << ToString() << " at " << split_point;
    S6_CHECK_GT(split_point, ranges_[index].startpoint)
        << "Zero-sized splits disallowed, splitting " << ToString() << " at "
        << split_point;
    ranges_[index].endpoint = split_point;
    ranges_.insert(ranges_.begin() + index + 1, new_range);

    return {LiveRange(this, index), LiveRange(this, index + 1)};
  }

  const Value* value_ = nullptr;
  // The range storage; this allows LiveRanges to not be invalidated.
  absl::InlinedVector<LiveRangeImpl, 8> ranges_;
  // Use points where this value is used, in order. A use point has an optional
  // location requirement.
  absl::InlinedVector<std::pair<int64_t, LocationRequirement>, 4> use_points_;
  // The chained live intervals, which can be nullptr.
  LiveInterval* incoming_chain_ = nullptr;
  LiveInterval* outgoing_chain_ = nullptr;
};

inline int64_t LiveRange::startpoint() const {
  return interval_->ranges_[index_].startpoint;
}

inline int64_t LiveRange::endpoint() const {
  return interval_->ranges_[index_].endpoint;
}

inline const Location& LiveRange::location() const {
  return interval_->ranges_[index_].location;
}

inline void LiveRange::set_location(const Location& loc) {
  interval_->ranges_[index_].location = loc;
}

inline absl::optional<LiveRange> LiveRange::Next() const {
  if (index_ + 1 < interval_->ranges_.size()) {
    return LiveRange(interval_, index_ + 1);
  }
  return {};
}

inline std::pair<LiveRange, LiveRange> LiveRange::Split(int64_t split_point) {
  return interval_->SplitRange(index_, split_point);
}

inline const LocationRequirement& LiveRange::hint() const {
  return interval_->ranges_[index_].hint;
}

inline void LiveRange::set_hint(const LocationRequirement& loc) {
  interval_->ranges_[index_].hint = loc;
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_LIVE_INTERVAL_H_
