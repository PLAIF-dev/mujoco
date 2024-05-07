// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_SDF_SQSOCKETSHELL_H_
#define MUJOCO_PLUGIN_SDF_SQSOCKETSHELL_H_

#include <optional>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include "sdf.h"

namespace mujoco::plugin::sdf {

struct SqSocketShellAttribute {
  static constexpr int nattribute = 8;
  static constexpr char const* names[nattribute] = {
    "outbox_x", "outbox_y", "outbox_z", "outbox_rounding", 
    "inbox_x", "inbox_y", "inbox_z", "inbox_rounding", 
    };
  static constexpr mjtNum defaults[nattribute] = { 0.028, 0.033, 0.028, 0.004, 0.0245, 0.033, 0.0245, 0.002};
};

class SqSocketShell {
 public:
  // Creates a new SqSocketShell instance or returns null on failure.
  static std::optional<SqSocketShell> Create(const mjModel* m, mjData* d, int instance);
  SqSocketShell(SqSocketShell&&) = default;
  ~SqSocketShell() = default;

  mjtNum Distance(const mjtNum point[3]) const;
  void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

  static void RegisterPlugin();

  mjtNum attribute[SqSocketShellAttribute::nattribute];

 private:
  SqSocketShell(const mjModel* m, mjData* d, int instance);
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_SQSOCKETSHELL_H_
