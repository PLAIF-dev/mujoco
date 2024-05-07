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
#include "socket.h"

namespace mujoco::plugin::sdf {
namespace {

static float opOnion(const mjtNum d1, const mjtNum thickness) {
    return mju_abs(d1) - thickness;
}

inline mjtNum opSubtraction(mjtNum a, mjtNum b) {
    return mju_max(-a, b);
}


// BUG: when h > r, the cylinder is not capped
static float sdCappedCylinder(const mjtNum p[3], mjtNum h, mjtNum r) {
    mjtNum pxy[2] = {p[0], p[1]};
    mjtNum d[2] = {mju_norm(pxy, 2) - r, mju_abs(p[2]) - h};
    mjtNum d0_capped = mju_max(d[0], 0.0), d1_capped = mju_max(d[1], 0.0);
    return mju_min(mju_max(d[0], d[1]), 0.0) + mju_sqrt(d0_capped * d0_capped + d1_capped * d1_capped);
}


// port above shadertoy to mujoco
static mjtNum distance(const mjtNum p[3], const mjtNum attrs[8]) {
    mjtNum bh = attrs[0];
    mjtNum br = attrs[1];
    mjtNum uh = attrs[2];
    mjtNum uor = attrs[3];
    mjtNum uir = attrs[4];
    mjtNum shr = attrs[5];
    mjtNum shh = attrs[6];
    mjtNum sho = attrs[7];
    
    
    mjtNum bottomCylinder =  sdCappedCylinder(p, br, br); //sdCappedCylinder(p, bh, br);

    mjtNum p2[3] = {p[0], p[1], p[2]-bh};
    mjtNum upperOnionCylinder = mju_max(
        mju_max(
            opOnion(
                sdCappedCylinder(p2, uh, uor),
                uor - uir),
            -1.0),
        -1.0);
    mjtNum p3[3] = {p[0], p[1]  + sho, p[2] - bh};
    mjtNum socket_left = sdCappedCylinder(p3, shh + bh, shr);
    
    mjtNum p4[3] = {p[0] , p[1] - sho, p[2] - bh};
    mjtNum socket_right = sdCappedCylinder(p4, shh + bh, shr);


    mjtNum dist = opSubtraction(
      Union(socket_left, socket_right),
      Union(upperOnionCylinder, bottomCylinder)
    );

    return dist;
}


}  // namespace

// factory function
std::optional<Socket> Socket::Create(
    const mjModel* m, mjData* d, int instance) {
  if (
    CheckAttr("bottom_height", m, instance) && CheckAttr("bottom_radius", m, instance)
    && CheckAttr("upper_height", m, instance) && CheckAttr("upper_outer_radius", m, instance) 
    && CheckAttr("upper_inner_radius", m, instance) && CheckAttr("socket_hole_radius", m, instance) 
    && CheckAttr("socket_hole_height", m, instance) && CheckAttr("socket_hole_offset", m, instance)) {
    return Socket(m, d, instance);
  } else {
    mju_warning("Invalid radius1 or radius2 parameters in Socket plugin");
    return std::nullopt;
  }
}

// plugin constructor
Socket::Socket(const mjModel* m, mjData* d, int instance) {
  SdfDefault<SocketAttribute> defattribute;

  for (int i=0; i < SocketAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        SocketAttribute::names[i],
        mj_getPluginConfig(m, instance, SocketAttribute::names[i]));
  }
}

// sdf
mjtNum Socket::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void Socket::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
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
void Socket::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.socket";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = SocketAttribute::nattribute;
  plugin.attributes = SocketAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = Socket::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new Socket(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Socket*>(d->plugin_data[instance]);
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
        auto* sdf = reinterpret_cast<Socket*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<Socket*>(d->plugin_data[instance]);
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
        SdfDefault<SocketAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
