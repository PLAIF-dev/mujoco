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
#include "sqsocket.h"

namespace mujoco::plugin::sdf {
namespace {


static float opOnion(const mjtNum d1, const mjtNum thickness) {
    return mju_abs(d1) - thickness;
}


inline mjtNum opSubtraction(mjtNum a, mjtNum b) {
    return mju_max(-a, b);
}


static mjtNum sdCappedCylinder(const mjtNum p[3], mjtNum h, mjtNum r) {
    mjtNum pxy[2] = {p[0], p[2]};
    mjtNum d[2] = {mju_norm(pxy, 2) - r, mju_abs(p[1]) - h};
    mjtNum d0_capped = mju_max(d[0], 0.0), d1_capped = mju_max(d[1], 0.0);
    return mju_min(mju_max(d[0], d[1]), 0.0) + mju_sqrt(d0_capped * d0_capped + d1_capped * d1_capped);
}


static mjtNum sdRoundBox(const mjtNum p[3], const mjtNum b[3], mjtNum r) {
    const mjtNum q[3] = {mju_abs(p[0]) - b[0], mju_abs(p[1]) - b[1], mju_abs(p[2]) - b[2]};
    const mjtNum cappedQ[3] = {mju_max(q[0], 0.0), mju_max(q[1], 0.0), mju_max(q[2], 0.0)} ;
    return mju_norm3(cappedQ) + mju_min(mju_max(q[0], mju_max(q[1], q[2])), 0.0);
}


static mjtNum distance(const mjtNum p[3], const mjtNum attrs[11]) {
    mjtNum box_x = attrs[0];
    mjtNum box_y = attrs[1];
    mjtNum box_z = attrs[2];
    mjtNum box_rounding = attrs[3];
    mjtNum hole_xpos = attrs[4];
    mjtNum hole_ypos = attrs[5];
    mjtNum hole_height = attrs[6];
    mjtNum hole_radius = attrs[7];
    mjtNum sqsocket_conn_radius = attrs[8];
    mjtNum sqsocket_conn_height = attrs[9];
    mjtNum sqsocket_conn_offset = attrs[10];
    
    mjtNum box_offset[3] = {box_x, box_y, box_z};
    mjtNum body_box =  sdRoundBox(p, box_offset, box_rounding); //sdCappedCylinder(p, bh, br);

    mjtNum hole_pos[3] = {p[0] - hole_xpos, p[1] - box_y, p[2] - hole_ypos};
    mjtNum hole = sdCappedCylinder(hole_pos, hole_height, hole_radius);
    
    mjtNum left_conn_pos[3] = {hole_pos[0], hole_pos[1] + hole_height, hole_pos[2] + sqsocket_conn_offset};
    mjtNum sqsocket_left = sdCappedCylinder(left_conn_pos, sqsocket_conn_height, sqsocket_conn_radius);
    
    mjtNum right_conn_pos[3] = {hole_pos[0] , hole_pos[1] + hole_height, hole_pos[2] - sqsocket_conn_offset};
    mjtNum sqsocket_right = sdCappedCylinder(right_conn_pos, sqsocket_conn_height, sqsocket_conn_radius);

    mjtNum dist = Subtraction(body_box, Union(hole, Union(sqsocket_left, sqsocket_right)));
    
    return dist;
}


}  // namespace


// factory function
std::optional<SqSocket> SqSocket::Create(
    const mjModel* m, mjData* d, int instance) {
  if (
    CheckAttr("box_x", m, instance) && CheckAttr("box_y", m, instance) 
    && CheckAttr("box_z", m, instance) && CheckAttr("box_rounding", m, instance) 
    && CheckAttr("hole_xpos", m, instance) && CheckAttr("hole_ypos", m, instance) 
    && CheckAttr("hole_height", m, instance) && CheckAttr("hole_radius", m, instance)
    && CheckAttr("sqsocket_conn_radius", m, instance) && CheckAttr("sqsocket_conn_height", m, instance)
    && CheckAttr("sqsocket_conn_offset", m, instance)
  ) {
    return SqSocket(m, d, instance);
  } else {
    mju_warning("Invalid radius1 or radius2 parameters in SqSocket plugin");
    return std::nullopt;
  }
}

// plugin constructor
SqSocket::SqSocket(const mjModel* m, mjData* d, int instance) {
  SdfDefault<SqSocketAttribute> defattribute;

  for (int i=0; i < SqSocketAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        SqSocketAttribute::names[i],
        mj_getPluginConfig(m, instance, SqSocketAttribute::names[i]));
  }
}

// sdf
mjtNum SqSocket::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void SqSocket::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
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
void SqSocket::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.sqsocket";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = SqSocketAttribute::nattribute;
  plugin.attributes = SqSocketAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = SqSocket::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new SqSocket(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<SqSocket*>(d->plugin_data[instance]);
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
        auto* sdf = reinterpret_cast<SqSocket*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<SqSocket*>(d->plugin_data[instance]);
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
        SdfDefault<SqSocketAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
