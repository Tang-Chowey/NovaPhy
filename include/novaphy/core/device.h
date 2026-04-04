#pragma once

namespace novaphy {

/**
 * @brief Device types for compute backend selection.
 *
 * @details Specifies which hardware backend executes simulation kernels.
 * `CPU` is always available. GPU backends require the corresponding driver
 * and runtime libraries to be installed.
 */
enum class DeviceType {
    CPU = 0,   /**< Host CPU execution (always available). */
    CUDA,      /**< NVIDIA GPU via CUDA. */
    Vulkan,    /**< Cross-vendor GPU via Vulkan compute. */
    Metal,     /**< Apple GPU via Metal. */
};

/**
 * @brief Lightweight device descriptor pairing a backend type with an ordinal.
 *
 * @details The ordinal selects among multiple devices of the same type
 * (e.g. multi-GPU systems). Ordinal 0 is the default device.
 */
struct Device {
    DeviceType type = DeviceType::CPU;  /**< Compute backend. */
    int ordinal = 0;                    /**< Device index within the backend. */

    /** @brief Convenience factory for the default CPU device. */
    static Device cpu() { return {DeviceType::CPU, 0}; }

    /** @brief Convenience factory for the default CUDA device. */
    static Device cuda(int ordinal = 0) { return {DeviceType::CUDA, ordinal}; }

    bool operator==(const Device& other) const {
        return type == other.type && ordinal == other.ordinal;
    }
    bool operator!=(const Device& other) const { return !(*this == other); }
};

}  // namespace novaphy
