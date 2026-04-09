#pragma once

#include <string>
#include "novaphy/math/math_types.h"

namespace novaphy {

struct Site {
    std::string label;
    Transform local_transform = Transform::identity();

    int body_index = -1;
    int articulation_index = -1;
    int link_index = -1;

    bool is_articulation_site() const {
        return articulation_index >= 0;
    }
};

}  // namespace novaphy
