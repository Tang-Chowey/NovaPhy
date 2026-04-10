#pragma once

#include <string>

#include "novaphy/novaphy_config.h"

namespace novaphy {

/**
 * @brief Returns the semantic version string of the NovaPhy library.
 * @return Version string in the form `major.minor.patch`.
 */
inline std::string version() {
    return NOVAPHY_VERSION;
}

}  // namespace novaphy
