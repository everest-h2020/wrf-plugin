#include "ABI.h"
#include "fxx/Memory.h"

#include <cmath>
#include <stdexcept>

using namespace fxx;
using namespace std;

namespace {

struct ReferenceProfiles {
    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    constexpr ReferenceProfiles(
        REAL T_min,
        REAL T_delta,
        REAL p_max,
        REAL p_log_delta,
        REAL p_tropo,
        tensor<REAL, 3> eta_half) noexcept
            : m_T_min(T_min),
              m_T_delta(T_delta),
              m_p_max(p_max),
              m_p_log_delta(p_log_delta),
              m_p_tropo(p_tropo),
              m_eta_half(std::move(eta_half))
    {
        // Used as divisors, and therefore may not be 0.
        assert(m_T_delta != 0);
        assert(m_p_log_delta != 0);
    }

    [[nodiscard]] static ReferenceProfiles load(
        memref<const REAL, 1> T_ref,
        memref<const REAL, 1> p_ref,
        REAL p_tropo,
        memref<const INTEGER, 2> flav_to_abs,
        memref<const REAL, 3> r_ref);

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr REAL T_min() const noexcept { return m_T_min; }
    [[nodiscard]] constexpr REAL T_delta() const noexcept { return m_T_delta; }
    [[nodiscard]] constexpr REAL p_max() const noexcept { return m_p_max; }
    [[nodiscard]] constexpr REAL p_log_delta() const noexcept
    {
        return m_p_log_delta;
    }
    [[nodiscard]] constexpr REAL p_tropo() const noexcept { return m_p_tropo; }
    [[nodiscard]] constexpr const tensor<REAL, 3> &eta_half() const noexcept
    {
        return m_eta_half;
    }

    //===------------------------------------------------------------------===//
    // Interpolation
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr REAL unmap_T(REAL f_T) const noexcept
    {
        return T_min() + T_delta() * f_T;
    }
    [[nodiscard]] constexpr REAL unmap_p(REAL f_p) const noexcept
    {
        return p_max() * std::exp(p_log_delta() * f_p);
    }

    [[nodiscard]] constexpr REAL map_T(REAL T) const noexcept
    {
        return (T - T_min()) / T_delta();
    }
    [[nodiscard]] constexpr REAL map_p(REAL p) const noexcept
    {
        return std::log(p / p_max()) / p_log_delta();
    }

    [[nodiscard]] void interpolate(
        memref<const REAL, 1> T_lay,
        memref<const REAL, 1> p_lay,
        memref<const REAL, 1> n_prime_d,
        memref<const INTEGER, 2> flav_to_abs,
        memref<const REAL, 2> r_lay,
        memref<REAL, 1> T_bar,
        memref<REAL, 1> p_bar,
        memref<REAL, 3> eta) const noexcept;

private:
    REAL m_T_min;
    REAL m_T_delta;
    REAL m_p_max;
    REAL m_p_log_delta;
    REAL m_p_tropo;
    tensor<REAL, 3> m_eta_half;
};

ReferenceProfiles ReferenceProfiles::load(
    memref<const REAL, 1> T_ref,
    memref<const REAL, 1> p_ref,
    REAL p_tropo,
    memref<const INTEGER, 2> flav_to_abs,
    memref<const REAL, 3> r_ref)
{
    // Recover temperature curve parameters.
    const auto n_T = T_ref.layout().hrect().sizes()[0];
    if (n_T <= 1) throw new runtime_error("Empty temperature reference.");
    const REAL T_min = T_ref(0);
    const REAL T_delta = (T_ref(n_T - 1) - T_min) / (n_T - 1);
    if (T_delta == 0) throw new runtime_error("Invalid temperature reference.");

    // Recover pressure curve parameters.
    const auto n_p = p_ref.layout().hrect().sizes()[0];
    if (n_p <= 1) throw new runtime_error("Empty pressure reference.");
    const REAL p_max = p_ref(0);
    const REAL p_log_delta =
        (std::log(p_ref(n_p - 1)) - std::log(p_max)) / (n_p - 1);
    if (T_delta == 0) throw new runtime_error("Invalid pressure reference.");

    // Recover the binary species parameter reference state.
    const auto n_flav = flav_to_abs.layout().hrect().sizes()[0];
    tensor<REAL, 3> eta_half(n_flav, 2, n_T);
    for (index_t i_flav = 0; i_flav < n_flav; ++i_flav)
        for (index_t i_layer = 0; i_layer < 2; ++i_layer)
            for (index_t i_temp = 0; i_temp < n_T; ++i_temp) {
                const auto r_1 = r_ref(i_temp, flav_to_abs(i_flav, 0), i_layer);
                const auto r_2 = r_ref(i_temp, flav_to_abs(i_flav, 1), i_layer);
                eta_half(i_flav, i_layer, i_temp) = r_1 / r_2;
            }

    return ReferenceProfiles(
        T_min,
        T_delta,
        p_max,
        p_log_delta,
        p_tropo,
        std::move(eta_half));
}

void ReferenceProfiles::interpolate(
    memref<const REAL, 1> T_lay,
    memref<const REAL, 1> p_lay,
    memref<const REAL, 1> n_prime_d,
    memref<const INTEGER, 2> flav_to_abs,
    memref<const REAL, 2> r_lay,
    memref<REAL, 1> T_bar,
    memref<REAL, 1> p_bar,
    memref<REAL, 3> eta) const noexcept
{
    constexpr REAL tiny = std::numeric_limits<REAL>::epsilon() * 2;

    // Shape checking.
    const auto n_lay = T_lay.layout().hrect().sizes()[0];
    assert(p_lay.layout().hrect().sizes()[0] == n_lay);
    assert(n_prime_d.layout().hrect().sizes()[0] == n_lay);
    assert(r_lay.layout().hrect().sizes()[0] == n_lay);
    assert(T_bar.layout().hrect().sizes()[0] == n_lay);
    assert(p_bar.layout().hrect().sizes()[0] == n_lay);
    const auto n_flav = eta_half().layout().hrect().sizes()[0];
    assert(flav_to_abs.layout().hrect().sizes()[0] == n_flav);
    assert(flav_to_abs.layout().hrect().sizes()[1] == 2);
    assert(eta.layout().hrect().sizes()[0] == n_lay);
    assert(eta.layout().hrect().sizes()[1] == n_flav);
    assert(eta.layout().hrect().sizes()[2] == 2);

    for (index_t i_lay = 0; i_lay < n_lay; ++i_lay) {
        const auto f_T = T_bar(i_lay) = map_T(T_lay(i_lay));
        const auto i_T = static_cast<index_t>(std::floor(f_T));
        const auto p = p_lay(i_lay);
        const auto is_strato = p < p_tropo();
        p_bar(i_lay) = map_p(p);

        for (index_t i_flav = 0; i_flav < n_flav; ++i_flav) {
            const auto r_1 = r_lay(i_lay, flav_to_abs(i_flav, 0));
            const auto r_2 = r_lay(i_lay, flav_to_abs(i_flav, 1));
            for (index_t i_temp = 0; i_temp < 2; ++i_temp) {
                const auto r_eta = eta_half()(i_flav, is_strato, i_temp);
                const auto r_mix = r_1 + r_eta * r_2;
                eta(i_lay, i_flav, i_temp) =
                    (r_mix >= tiny ? (r_1 / r_mix) : 0.5) * (n_eta - 1);
            }
        }
    }
}

} // namespace
