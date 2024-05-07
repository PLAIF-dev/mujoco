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

#ifndef MUJOCO_PLUGIN_SDF_SOCKET_H_
#define MUJOCO_PLUGIN_SDF_SOCKET_H_

#include <optional>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include "sdf.h"

namespace mujoco::plugin::sdf {

struct SocketAttribute {
  static constexpr int nattribute = 8;
  static constexpr char const* names[nattribute] = {
    "bottom_height", "bottom_radius", 
    "upper_height", "upper_outer_radius", "upper_inner_radius",
    "socket_hole_radius", "socket_hole_height", "socket_hole_offset",
    };
  static constexpr mjtNum defaults[nattribute] = { .024, .0175, .011, .0217, 0.0195, 0.0027, 0.008, 0.009};
};

class Socket {
 public:
  // Creates a new Socket instance or returns null on failure.
  static std::optional<Socket> Create(const mjModel* m, mjData* d, int instance);
  Socket(Socket&&) = default;
  ~Socket() = default;

  mjtNum Distance(const mjtNum point[3]) const;
  void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

  static void RegisterPlugin();

  mjtNum attribute[SocketAttribute::nattribute];

 private:
  Socket(const mjModel* m, mjData* d, int instance);
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_SOCKET_H_
