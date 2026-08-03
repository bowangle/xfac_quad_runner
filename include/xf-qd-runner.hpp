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

#include <nlohmann/json.hpp>

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

    // Save real values as strings so JSON parsing does not narrow
    // dd_128 or float128 values through double.
    std::string to_string_for_save(const std::vector<Real>& v)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            oss << "\"" << to_string(v[i]) << "\"";
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

    static std::string double_to_exact_string(double value)
    {
        std::ostringstream oss;
        oss << std::scientific
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << value;
        return oss.str();
    }

    static void write_dd128_exact_real(std::ostream& out, const dd_128& value)
    {
        out << "{\"hi\": \"" << double_to_exact_string(value.x[0])
            << "\", \"lo\": \"" << double_to_exact_string(value.x[1])
            << "\"}";
    }

    static void write_dd128_exact_complex(std::ostream& out,
                                          const Cdd_128& value)
    {
        out << "{\"real\": ";
        write_dd128_exact_real(out, value.real());
        out << ", \"imag\": ";
        write_dd128_exact_real(out, value.imag());
        out << "}";
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
            << to_string_for_save(l_discontinuity) << ",\n";
        file << "  \"l_f_discontinuity\": "
            << to_string_for_save(l_f_discontinuity);

        if constexpr (std::is_same_v<Real, dd_128>) {
            file << ",\n  \"dd128_exact\": {\n";

            file << "    \"value\": ";
            write_dd128_exact_complex(file, value);
            file << ",\n";

            file << "    \"l_discontinuity\": [";
            for (std::size_t i = 0; i < l_discontinuity.size(); ++i) {
                if (i != 0) file << ", ";
                write_dd128_exact_real(file, l_discontinuity[i]);
            }
            file << "],\n";

            file << "    \"l_f_discontinuity\": [";
            for (std::size_t i = 0; i < l_f_discontinuity.size(); ++i) {
                if (i != 0) file << ", ";
                write_dd128_exact_complex(file, l_f_discontinuity[i]);
            }
            file << "]\n  }";
        }

        file << "\n}\n";
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


// =========================================================================
// Free functions for loading f_value JSON data saved by save_to_json()
// =========================================================================

/// Parse directly into Real so high-precision values never pass through
/// double or long double. This works with the stream extractors provided for
/// double, dd_128, and boost::multiprecision::float128.
template <typename Real>
Real parse_real_str(const std::string& s)
{
    std::istringstream input(s);
    Real value;
    input >> value;
    if (!input) {
        throw std::runtime_error("parse_real_str: invalid value \"" + s + "\"");
    }

    input >> std::ws;
    if (!input.eof()) {
        throw std::runtime_error(
            "parse_real_str: trailing characters in \"" + s + "\"");
    }
    return value;
}

/// Parse a complex number from a string produced by to_string(), e.g.
/// "-5.00e-01+8.66e-01j" or "3.14e+00-2.72e+00j".
/// Also handles the parenthesised form from to_string_for_save(),
/// e.g. "(-5.00e-01+8.66e-01j)".
template <typename Complex>
Complex parse_complex_str(const std::string& s)
{
    using Real = typename Complex::value_type;

    std::string str = s;

    // Strip optional wrapping parentheses (from to_string_for_save)
    if (!str.empty() && str.front() == '(' && str.back() == ')') {
        str = str.substr(1, str.size() - 2);
    }

    // Strip trailing 'j'
    if (!str.empty() && str.back() == 'j') {
        str.pop_back();
    }

    // Find the sign that separates real and imaginary parts.
    // It is the last '+' or '-' whose predecessor is not 'e'/'E'
    // (to avoid matching the exponent sign in scientific notation).
    std::size_t split_pos = std::string::npos;
    for (std::size_t i = str.size() - 1; i > 0; --i) {
        if ((str[i] == '+' || str[i] == '-') &&
            str[i - 1] != 'e' && str[i - 1] != 'E')
        {
            split_pos = i;
            break;
        }
    }

    if (split_pos == std::string::npos) {
        throw std::runtime_error(
            "parse_complex_str: cannot split real/imag in \"" + s + "\"");
    }

    Real re = parse_real_str<Real>(str.substr(0, split_pos));
    Real im = parse_real_str<Real>(str.substr(split_pos));
    return Complex(re, im);
}

inline dd_128 load_dd128_exact_real(const nlohmann::json& j)
{
    const double hi = parse_real_str<double>(j.at("hi").get<std::string>());
    const double lo = parse_real_str<double>(j.at("lo").get<std::string>());
    return dd_128(dd_real(hi, lo));
}

inline Cdd_128 load_dd128_exact_complex(const nlohmann::json& j)
{
    return Cdd_128(load_dd128_exact_real(j.at("real")),
                    load_dd128_exact_real(j.at("imag")));
}

/// Load f_value data from a JSON file previously written by
/// TCI2_1D_Runner::save_to_json().
/// Returns {value, l_discontinuity, l_f_discontinuity}.
template <typename Complex>
std::tuple<Complex,
           std::vector<typename Complex::value_type>,
           std::vector<Complex>>
load_fvalues_from_json(const std::string& filename)
{
    using Real = typename Complex::value_type;

    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("load_fvalues_from_json: cannot open \""
                                 + filename + "\"");
    }

    nlohmann::json j;
    file >> j;

    if constexpr (std::is_same_v<Real, dd_128>) {
        // New dd_128 files carry an exact representation of both underlying
        // doubles. Older files have no such section and use the decimal
        // compatibility path below.
        if (j.contains("dd128_exact")) {
            const auto& exact = j.at("dd128_exact");

            Complex value = load_dd128_exact_complex(exact.at("value"));

            std::vector<Real> l_discontinuity;
            for (const auto& x : exact.at("l_discontinuity")) {
                l_discontinuity.push_back(load_dd128_exact_real(x));
            }

            std::vector<Complex> l_f_discontinuity;
            for (const auto& x : exact.at("l_f_discontinuity")) {
                l_f_discontinuity.push_back(load_dd128_exact_complex(x));
            }

            return {value, l_discontinuity, l_f_discontinuity};
        }
    }

    // Decimal compatibility representation, used for double/float128 and
    // when loading old files. A double target loading a dd_128 file also takes
    // this path rather than reading dd128_exact.
    Complex value = parse_complex_str<Complex>(j.at("value").get<std::string>());

    std::vector<Real> l_discontinuity;
    for (const auto& x : j.at("l_discontinuity")) {
        if (x.is_string()) {
            l_discontinuity.push_back(
                parse_real_str<Real>(x.get<std::string>()));
        } else {
            // Backward compatibility with older files that stored JSON
            // numbers. Their precision may already have been narrowed by
            // the JSON parser when Real is wider than double.
            l_discontinuity.push_back(
                parse_real_str<Real>(x.dump()));
        }
    }

    std::vector<Complex> l_f_discontinuity;
    for (const auto& s : j.at("l_f_discontinuity")) {
        l_f_discontinuity.push_back(
            parse_complex_str<Complex>(s.get<std::string>()));
    }

    return {value, l_discontinuity, l_f_discontinuity};
}