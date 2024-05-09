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

#ifndef MUJOCO_PLUGIN_SDF_SQSOCKET_H_
#define MUJOCO_PLUGIN_SDF_SQSOCKET_H_

#include <optional>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include "sdf.h"

namespace mujoco::plugin::sdf {

struct SqSocketAttribute {
  static constexpr int nattribute = 11;
  static constexpr char const* names[nattribute] = {
    "box_x", "box_y", "box_z", "box_rounding", 
    "hole_xpos", "hole_ypos", "hole_height", "hole_radius",
    "sqsocket_conn_radius", "sqsocket_conn_height", "sqsocket_conn_offset",
    };
  static constexpr mjtNum defaults[nattribute] = { .0245, .033, .0245, .003, 0.0, 0.0, 0.009, 0.0195, 0.0024, 0.02, 0.009};
};

class SqSocket {
 public:
  // Creates a new SqSocket instance or returns null on failure.
  static std::optional<SqSocket> Create(const mjModel* m, mjData* d, int instance);
  SqSocket(SqSocket&&) = default;
  ~SqSocket() = default;

  mjtNum Distance(const mjtNum point[3]) const;
  void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

  static void RegisterPlugin();

  mjtNum attribute[SqSocketAttribute::nattribute];

 private:
  SqSocket(const mjModel* m, mjData* d, int instance);
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_SQSOCKET_H_
