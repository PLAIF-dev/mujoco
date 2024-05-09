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

#include <cstdint>
#include <optional>
#include <utility>
#include <iostream>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "sqsocketshell.h"

namespace mujoco::plugin::sdf {
namespace {


inline mjtNum opSubtraction(mjtNum a, mjtNum b) {
  return mju_max(-a, b);
}


static mjtNum sdRoundBox(const mjtNum p[3], const mjtNum b[3], mjtNum r) {
  const mjtNum q[3] = {mju_abs(p[0]) - b[0], mju_abs(p[1]) - b[1], mju_abs(p[2]) - b[2]};
  const mjtNum cappedQ[3] = {mju_max(q[0], 0.0), mju_max(q[1], r), mju_max(q[2], 0.0)} ;
  return mju_norm3(cappedQ) + mju_min(mju_max(q[0], mju_max(q[1], q[2])), 0.0) - r;
}


static mjtNum distance(const mjtNum p[3], const mjtNum attrs[8]) {
  // All box lengths are half, as they extend in both directions from the origin
  mjtNum ox = attrs[0];  // Outer box x width
  mjtNum oy = attrs[1];  // Outer box y height 
  mjtNum oz = attrs[2];  // Outer box z depth
  mjtNum or_ = attrs[3];  // Outer box rounding
  mjtNum ix = attrs[4];  // Inner box x width
  mjtNum iy = attrs[5];  // Inner box y height
  mjtNum iz = attrs[6];  // Inner box z depth
  mjtNum ir = attrs[7];  // Inner box rounding
  
  mjtNum outboxOffset[3] = {ox, oy, oz};
  mjtNum outbox = sdRoundBox(p, outboxOffset, or_); //sdCappedCylinder(p, bh, br);

  mjtNum inboxOffset[3] = {ix, iy, iz};
  mjtNum inbox = sdRoundBox(p, inboxOffset, ir); //sdCappedCylinder(p, bh, br);

  mjtNum dist = Subtraction(outbox, inbox);
  
  return dist;
}


}  // namespace


// factory function
std::optional<SqSocketShell> SqSocketShell::Create(
    const mjModel* m, mjData* d, int instance) {
  if (
    CheckAttr("outbox_x", m, instance) && CheckAttr("outbox_y", m, instance) 
    && CheckAttr("outbox_z", m, instance) && CheckAttr("outbox_rounding", m, instance)
    && CheckAttr("inbox_x", m, instance) && CheckAttr("inbox_y", m, instance)
    && CheckAttr("inbox_z", m, instance) && CheckAttr("inbox_rounding", m, instance)
  ) {
    return SqSocketShell(m, d, instance);
  } else {
    mju_warning("Invalid radius1 or radius2 parameters in SqSocketShell plugin");
    return std::nullopt;
  }
}

// plugin constructor
SqSocketShell::SqSocketShell(const mjModel* m, mjData* d, int instance) {
  SdfDefault<SqSocketShellAttribute> defattribute;

  for (int i=0; i < SqSocketShellAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        SqSocketShellAttribute::names[i],
        mj_getPluginConfig(m, instance, SqSocketShellAttribute::names[i]));
  }
}

// sdf
mjtNum SqSocketShell::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void SqSocketShell::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
  mjtNum eps = 1e-8;
  mjtNum dist0 = distance(point, attribute);

  mjtNum pointX[3] = {point[0]+eps, point[1], point[2]};
  mjtNum distX = distance(pointX, attribute);
  mjtNum pointY[3] = {point[0], point[1]+eps, point[2]};
  mjtNum distY = distance(pointY, attribute);
  mjtNum pointZ[3] = {point[0], point[1], point[2]+eps};
  mjtNum distZ = distance(pointZ, attribute);

  grad[0] = (distX - dist0) / eps;
  grad[1] = (distY - dist0) / eps;
  grad[2] = (distZ - dist0) / eps;
}

// plugin registration
void SqSocketShell::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.sqsocketshell";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = SqSocketShellAttribute::nattribute;
  plugin.attributes = SqSocketShellAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = SqSocketShell::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new SqSocketShell(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<SqSocketShell*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, double* plugin_state, void* plugin_data,
                     int instance) {
    // do nothing
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        // do nothing;
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<SqSocketShell*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<SqSocketShell*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = aabb[4] = attributes[0] + attributes[1];
        aabb[5] = attributes[1];
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<SqSocketShellAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
