#pragma once
#include <Eigen/Dense>
#include <xfac_quad/xfac_quad.hpp>   // TensorCI2Param
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <ostream>
#include <utility>
#include <cstddef>
#include <stdexcept>   // std::invalid_argument

/// Which caching layer to use. The two layers are mutually exclusive, so this
/// makes the illegal combination unrepresentable rather than merely checked.
enum class CacheLevel {
    none,       ///< no caching at all
    xfac,       ///< internal xfac caching, avoids function calls to the same point
    runner      ///< external caching at the runner level
};

namespace xf_qd_detail {

    inline const char* cache_name(CacheLevel c) {
        switch (c) {
            case CacheLevel::none:   return "none";
            case CacheLevel::xfac:   return "xfac";
            case CacheLevel::runner: return "runner";
        }
        return "unknown";
    }

    inline std::string vec_to_string(const std::vector<int>& v) {
        if (v.empty()) return "(none)";
        std::ostringstream os;
        os << "[";
        for (std::size_t i = 0; i < v.size(); ++i)
            os << (i ? ", " : "") << v[i];
        os << "]";
        return os.str();
    }
} // namespace xf_qd_detail

/// Optional settings for TCI2_1D_runner: everything with a sensible default.
/// Set via designated initializers, e.g. { .fullPiv = true, .nRookIter = 5 }.
/// Designators must appear in declaration order.
template<class T>
struct TCI2_1D_runner_opts {
    using Real = typename Eigen::NumTraits<T>::Real;

    Real reltol = Eigen::NumTraits<Real>::epsilon() * Real(100);
                                                ///< CI will stop when pivotError < reltol*max(pivotError)
    std::vector<int> pivot1 = {};               ///< the first pivot (optional)
    bool fullPiv = false;                       ///< whether to search in the full Pi matrix to find a new pivot
    int nRookIter = 3;                          ///< number of rook pivoting iterations, used when fullPiv=false
    CacheLevel cache = CacheLevel::xfac;        ///< which caching layer to use, see CacheLevel
};

/// Parameters for TCI2_1D_runner. The three grid/iteration sizes are mandatory
/// and positional; everything else goes through `opts`.
///
/// Minimal: TCI2_1D_runner_param<double> p{20, 100, 50};
/// With options, designators must appear in declaration order:
///   TCI2_1D_runner_param<double> p{20, 100, 50, {.fullPiv = true, .cache = CacheLevel::runner}};
template<class T>
struct TCI2_1D_runner_param {
    using Real = typename Eigen::NumTraits<T>::Real;

    int nBit;                                   ///< number of bit on the 1D grid
    int nb_iter;                                ///< max number of iterate
    int bondDim;                                ///< max bond dimension of the tensor train

    // Owned copies of the optional settings (defaults come from TCI2_1D_runner_opts).
    Real reltol;                                ///< CI will stop when pivotError < reltol*max(pivotError)
    std::vector<int> pivot1;                    ///< the first pivot (empty = none given)
    bool fullPiv;                               ///< whether to search in the full Pi matrix to find a new pivot
    int nRookIter;                              ///< number of rook pivoting iterations, used when fullPiv=false
    CacheLevel cache;                           ///< which caching layer to use, see CacheLevel

    // Derived from `cache`: the flags the two layers actually read.
    // Mutually exclusive by construction.
    bool useCachedFunction_xfac_lvl;            ///< internal xfac caching, avoids function calls to the same point
    bool do_cache_runner_lvl;                   ///< external caching at the runner level

    TCI2_1D_runner_param(int nBit_,
                         int nb_iter_,
                         int bondDim_,
                         TCI2_1D_runner_opts<T> opts = {})
        : nBit(nBit_),
          nb_iter(nb_iter_),
          bondDim(bondDim_),
          reltol(opts.reltol),
          pivot1(std::move(opts.pivot1)),
          fullPiv(opts.fullPiv),
          nRookIter(opts.nRookIter),
          cache(opts.cache),
          useCachedFunction_xfac_lvl(cache == CacheLevel::xfac),
          do_cache_runner_lvl(cache == CacheLevel::runner)
    {
        if (!pivot1.empty() &&
            pivot1.size() != static_cast<std::size_t>(nBit))
        {
            std::ostringstream os;
            os << "TCI2_1D_runner_param: pivot1 has " << pivot1.size()
               << " entries but nBit = " << nBit
               << "; pivot1 must either be empty or have exactly nBit entries.";
            throw std::invalid_argument(os.str());
        }
    }

    /// Build the xfac-level parameters from this runner-level configuration.
    /// Fields TensorCI2Param owns but the runner does not expose (weight, cond)
    /// keep their defaults.
    xfac_quad::TensorCI2Param<T> toTensorCI2Param() const {
        return {
            .bondDim           = bondDim,
            .reltol            = reltol,
            .pivot1            = pivot1,
            .fullPiv           = fullPiv,
            .nRookIter         = nRookIter,
            .weight            = {},
            .cond              = {},
            .useCachedFunction = useCachedFunction_xfac_lvl
        };
    }

    /// Human-readable dump of the configuration, for log(p.to_string()).
    std::string to_string() const {
        std::ostringstream os;
        os << std::scientific
           << std::setprecision(Eigen::NumTraits<Real>::digits10() + 3);
        os << "TCI2_1D_runner_param:"
           << "\n  nBit        = " << nBit
           << "\n  nb_iter     = " << nb_iter
           << "\n  bondDim     = " << bondDim
           << "\n  reltol      = " << reltol
           << "\n  pivot1      = " << xf_qd_detail::vec_to_string(pivot1)
           << "\n  fullPiv     = " << (fullPiv ? "true" : "false")
           << "\n  nRookIter   = " << nRookIter
           << "\n  cache       = " << xf_qd_detail::cache_name(cache)
           << " (xfac_lvl = "   << (useCachedFunction_xfac_lvl ? "true" : "false")
           << ", runner_lvl = " << (do_cache_runner_lvl ? "true" : "false") << ")";
        return os.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const TCI2_1D_runner_param& p) {
        return os << p.to_string();
    }
};
// TCI2_1D_runner_param<double> p{20, 100, 50, {.fullPiv = true, .cache = CacheLevel::runner}};
// TCI2_1D_runner_param<double> p(20, 100, 50, {.fullPiv = true, .cache = CacheLevel::runner});