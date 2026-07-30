#pragma once

#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <array>
#include <stdexcept>

#include <complex>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <unordered_map>
#include <memory>
#include <type_traits>

#include <spdlog/spdlog.h>

#include <grid.h> // QTgrid_quad.
#include <xfac_quad/xfac_quad.hpp>

#include "xf-quad-param.hpp"
#include "xf-qd-output.hpp"

template <typename Scalar, typename Sint>
class TCI2_1D_Runner{
    public:

    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Complex = std::complex<Real>;

    const QTGrid<Real, Sint> grid;
    const TCI2_1D_runner_param<Scalar> tci_param;
    std::function<Scalar(Real)> function_x;               // original function: Scalar(Real)
    std::function<Scalar(std::vector<int>)> function_id;  // function on grid: Scalar(id)
    const std::vector<int> l_d;
    std::atomic<int> counter{0};
    std::atomic<int> counter_cached{0};
    std::shared_ptr<spdlog::logger> const logger;

    TCI2_1D_Runner(
        QTGrid<Real, Sint> grid_,
        TCI2_1D_runner_param<Scalar> tci_param_,
        std::function<Scalar(Real)> function_x_,
        std::shared_ptr<spdlog::logger> logger_ = nullptr
    )
    :
        grid(std::move(grid_)),
        tci_param(std::move(tci_param_)),
        function_x(std::move(function_x_)),
        // Uses the members `grid` and `function_x`, which are declared above and
        // therefore already initialised at this point. Never the constructor
        // parameters: those die when the constructor returns.
        function_id(resolve_function(tci_param.do_cache_runner_lvl)),
        l_d(std::vector<int>(grid.get_nBits(), 2)),
        logger(std::move(logger_))
    {
        if (grid.get_nBits() != tci_param.nBit) {
            std::ostringstream os;
            os << "TCI2_1D_Runner: grid has " << grid.get_nBits()
               << " bits but tci_param.nBit = " << tci_param.nBit
               << "; they must agree.";
            throw std::invalid_argument(os.str());
        }
    }

    // The cached/counting lambdas stored in `function_id` capture `this`, so a
    // copied or moved runner would hold callables pointing at the original.
    // (The std::atomic members already suppress these implicitly; spelling them
    // out documents why.)
    TCI2_1D_Runner(const TCI2_1D_Runner&)            = delete;
    TCI2_1D_Runner& operator=(const TCI2_1D_Runner&) = delete;
    TCI2_1D_Runner(TCI2_1D_Runner&&)                 = delete;
    TCI2_1D_Runner& operator=(TCI2_1D_Runner&&)      = delete;

    /// Wrap the coordinate-space function as a function on grid indices.
    /// Captures the grid *by value*, so the returned callable owns everything
    /// it needs and does not depend on any caller's lifetime.
    std::function<Scalar(std::vector<int>)> func_to_grid()
    {
        return [this, qt_grid = grid, f = function_x](const std::vector<int>& id) -> Scalar {
            counter.fetch_add(1, std::memory_order_relaxed);
            return f(qt_grid.id_to_coord(id));
        };
    }

    struct MultiIndexHash {
        size_t operator()(std::vector<int> const& mi) const noexcept {
            size_t seed = mi.size();
            for (int x : mi) {
                seed ^= static_cast<size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    static constexpr std::size_t kNShards = 64;

    // alignas(64) keeps neighbouring shards off the same cache line, so threads
    // working on different shards do not contend through false sharing.
    struct alignas(64) CacheShard {
        std::shared_mutex mtx;
        std::unordered_map<std::vector<int>, Scalar, MultiIndexHash> map;
    };

    std::function<Scalar(std::vector<int>)>
    make_cached_function(std::function<Scalar(std::vector<int>)> f)
    {
        auto shards = std::make_shared<std::array<CacheShard, kNShards>>();

        return [this, f = std::move(f), shards](const std::vector<int>& v) -> Scalar
        {
            CacheShard& s = (*shards)[MultiIndexHash{}(v) % kNShards];

            {   // read path — many threads at once
                std::shared_lock<std::shared_mutex> lk(s.mtx);
                auto it = s.map.find(v);
                if (it != s.map.end()) {
                    counter_cached.fetch_add(1, std::memory_order_relaxed);
                    return it->second;
                }
            }   // lock released here

            // Expensive call, deliberately made with no lock held. Two threads
            // may race to compute the same key; f is pure, so the duplicate
            // work is harmless and the first writer wins below.
            // `counter` is incremented inside func_to_grid, not here.
            Scalar result = f(v);

            {   // write path — one thread at a time, per shard
                std::unique_lock<std::shared_mutex> lk(s.mtx);
                return s.map.emplace(v, result).first->second;
            }
        };
    }

    std::function<Scalar(std::vector<int>)>
    resolve_function(bool do_cache)
    {
        std::function<Scalar(std::vector<int>)> function_on_grid = func_to_grid();
        if (do_cache){
            return make_cached_function(function_on_grid);
        }
        else{
            return function_on_grid;
        }
    }

    template <typename T>
    std::string to_string_stream(const T& v) {
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }

    // ---- to_string for real Scalar ----
    template <typename S = Scalar>
    std::enable_if_t<Eigen::NumTraits<S>::IsComplex == 0, std::string>
    to_string(const S& z)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<Real>::digits10 + 5) << std::scientific;
        oss << z;
        return oss.str();
    }

    // ---- to_string for complex Scalar ----
    template <typename S = Scalar>
    std::enable_if_t<Eigen::NumTraits<S>::IsComplex != 0, std::string>
    to_string(const S& z)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<Real>::digits10 + 5) << std::scientific;
        oss << z.real();
        if (z.imag() >= 0) oss << "+";
        oss << z.imag() << "j";
        return oss.str();
    }

    // ---- to_string for vector<Real> (only when Scalar is complex) ----
    template <typename S = Scalar>
    std::enable_if_t<Eigen::NumTraits<S>::IsComplex != 0, std::string>
    to_string(const std::vector<Real>& v)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<Real>::digits10 + 5) << std::scientific;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            oss << v[i];
            if (i + 1 < v.size()) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

    // ---- to_string for vector<Scalar> ----
    std::string to_string(const std::vector<Scalar>& v)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<Real>::digits10 + 5) << std::scientific;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            oss << v[i];
            if (i + 1 < v.size()) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

    // ---- to_string for vector<Complex> ----
    std::string to_string_for_save(const std::vector<Complex>& v)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            oss << "\"(" << to_string(v[i]) << ")\"";
            if (i + 1 < v.size()) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

    void save_to_json(
        const std::string& filename,
        const Complex& value,
        const std::vector<Real>& l_discontinuity,
        const std::vector<Complex>& l_f_discontinuity)
    {
        std::ofstream file(filename);
        file << "{\n";
        file << "  \"value\": \"" << to_string(value) << "\",\n";
        file << "  \"l_discontinuity\": "
            << to_string(l_discontinuity) << ",\n";
        file << "  \"l_f_discontinuity\": "
            << to_string_for_save(l_f_discontinuity) << "\n";
        file << "}\n";
    }

    // ---- helpers to convert Scalar <-> Complex ----
    static Complex to_complex(const Scalar& s) {
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex != 0)
            return s;
        else
            return Complex(s, Real(0));
    }

    static Scalar from_complex(const Complex& c) {
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex != 0)
            return c;
        else
            return c.real();
    }

    void fit(
        const std::vector<Real>& additional_pivot,
        bool verbose = true,
        bool do_save = false,
        const std::string& file_prefix="",
        int nb_point_res=1000,
        const std::vector<Real>& E_discontinuity={}
    ){
        auto log = [&](const std::string& msg) {
            if (logger) {
                logger->info(msg);
            }
            if (verbose) {
                std::cout << msg << "\n";
            }
        };

        // ===================== Resolve parameter =======================

        // pivot1 is allowed to be empty (the default: "no first pivot given"),
        // in which case there is no E_init to report.
        const bool has_pivot1 = !tci_param.pivot1.empty();
        const Real E_init = has_pivot1 ? grid.id_to_coord(tci_param.pivot1)
                                       : Real(0);

        std::vector<std::vector<int>> id_additionnal_pivot;
        id_additionnal_pivot.reserve(additional_pivot.size());
        for (const auto& E : additional_pivot)
            id_additionnal_pivot.push_back(grid.coord_to_id(E));

        // ===================== printing parameter ======================
        {
            log("========== FIT PARAMETERS ==========");

            if (has_pivot1) {
                log("E_init = " + to_string_stream(E_init));
                log("E_init id = " + xf_qd_detail::vec_to_string(tci_param.pivot1));
            } else {
                log("E_init = (none: pivot1 not given)");
            }

            std::ostringstream other_E_str;
            other_E_str << "[";
            for (size_t i = 0; i < additional_pivot.size(); ++i) {
                other_E_str << to_string_stream(additional_pivot[i]);
                if (i + 1 < additional_pivot.size()) other_E_str << ", ";
            }
            other_E_str << "]";

            std::ostringstream id_str;
            id_str << "[";
            for (size_t i = 0; i < id_additionnal_pivot.size(); ++i) {
                id_str << "[";
                for (size_t j = 0; j < id_additionnal_pivot[i].size(); ++j) {
                    id_str << id_additionnal_pivot[i][j];
                    if (j + 1 < id_additionnal_pivot[i].size()) id_str << ", ";
                }
                id_str << "]";
                if (i + 1 < id_additionnal_pivot.size()) id_str << ", ";
            }
            id_str << "]";

            log("additional_pivot = " + other_E_str.str());
            log("id_additionnal_pivot = " + id_str.str());

            log("verbose = " + to_string_stream(verbose ? "true" : "false"));
            log("do_save = " + to_string_stream(do_save ? "true" : "false"));
            log("file_prefix = " + file_prefix);
            log("nb_point_res = " + to_string_stream(nb_point_res));

            log(tci_param.to_string());
        }

        // ===================== optional parameter ======================

        if (!E_discontinuity.empty()){
            Scalar f_a = function_x(grid.get_a());
            Scalar f_b = function_x(grid.get_b());

            std::vector<Complex> l_f_discontinuity;
            l_f_discontinuity.reserve(E_discontinuity.size());
            for (const Real& e : E_discontinuity)
                l_f_discontinuity.push_back(to_complex(function_x(e)));

            log("  f_a = " + to_string(f_a));
            log("  f_b = " + to_string(f_b));
            log("  E_discontinuity = " + to_string(E_discontinuity));
            log("  l_f_discontinuity = " + to_string_for_save(l_f_discontinuity));
            save_to_json(
                file_prefix + "_f_values.json",
                to_complex(f_b),
                E_discontinuity,
                l_f_discontinuity
            );
        }

        // ===================== actual computation ======================

        log("==========begin iteration===============");

        std::vector<int> local_dim(tci_param.nBit, 2);

        xfac_quad::TensorFunction<Scalar> TF_function(function_id, tci_param.useCachedFunction_xfac_lvl);
        xfac_quad::TensorCI2<Scalar> tci(TF_function, local_dim, tci_param.toTensorCI2Param());

        tci.addPivotsAllBonds(id_additionnal_pivot);

        for (int it = 0; it < tci_param.nb_iter; ++it) {
            tci.iterate();

            std::ostringstream oss_pivot;
            oss_pivot << std::scientific
                    << std::setprecision(std::numeric_limits<Real>::max_digits10);
            oss_pivot << "sweep " << (it + 1)
                    << " |pivot error| = "
                    << tci.pivotError.back();
            log(oss_pivot.str());
        }

        xfac_quad::TensorTrain<Scalar> tt_temp = tci.tt;

        // save as complex (save_Tensor3D_to_arma works on complex only)
        {
            std::vector<xfac_quad::Tensor3D<Complex>> cores_complex;
            cores_complex.reserve(tt_temp.M.size());
            for (const auto& core : tt_temp.M) {
                xfac_quad::Tensor3D<Complex> ccore(core.n_rows, core.n_cols, core.n_slices);
                for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(core.data.size()); ++i)
                    ccore.data[i] = to_complex(core.data[i]);
                cores_complex.push_back(std::move(ccore));
            }
            std::ofstream tt_file(file_prefix + ".tt");
            xfac_quad::save_Tensor3D_to_arma(tt_file, cores_complex);
        }

        int max_bond_dim = 0;
        for (const auto& core : tt_temp.M)
            max_bond_dim = std::max(max_bond_dim, static_cast<int>(core.n_slices));
        log("Max bond dim: " + std::to_string(max_bond_dim));

        log("Nb function call:" + std::to_string(counter.load()));
        log("Nb function cached call:" + std::to_string(counter_cached.load()));

        // build complex TT for error computation
        xfac_quad::TensorTrain<Complex> tt_complex;
        tt_complex.M.reserve(tt_temp.M.size());
        for (const auto& core : tt_temp.M) {
            xfac_quad::Tensor3D<Complex> ccore(core.n_rows, core.n_cols, core.n_slices);
            for (Eigen::Index i = 0; i < core.n_rows * core.n_cols * core.n_slices; ++i)
                ccore.data[i] = to_complex(core.data[i]);
            tt_complex.M.push_back(std::move(ccore));
        }

        std::function<Complex(std::vector<int>)> gf_id_complex =
            [this](std::vector<int> id) -> Complex {
                return to_complex(function_id(std::move(id)));
            };

        TTErrorOnGrid<Real> error =
            error_TT_on_grid_point(tt_complex, gf_id_complex, grid, nb_point_res);

        std::ostringstream oss;
        oss << error;
        log(oss.str());

        if (do_save){
            save_TTErrorOnGrid(error, file_prefix + "_error.dat");
            grid.save_json(file_prefix);
        }
        // makeCanonical
    }

};