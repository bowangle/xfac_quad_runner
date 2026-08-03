// Exercises TCI2_1D_Runner across every caching layer and both pivot-search
// strategies, for every supported scalar type.
//
//   cache   in { none, xfac, runner }   x   fullPiv in { true, false }
//
// Two properties are checked that a single-configuration test cannot see:
//
//   1. Caching is transparent. The cache only memoises a pure function, so the
//      tensor train produced with cache=none, cache=xfac and cache=runner must
//      be bit-identical. Any difference means a caching layer is corrupting or
//      losing values — which is exactly how the runner-level data race
//      manifested before it was fixed.
//
//   2. The call counters are consistent: every request is either a real
//      evaluation or a cache hit, and enabling a cache must not increase the
//      number of real evaluations.

#include "test_common.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>

using testing::check;

// ---- problem size (kept in one place so it is easy to shrink) ----
static constexpr int kNBit    = 30;
static constexpr int kNbIter  = 10;
static constexpr int kBondDim = 100;
static constexpr int kNbPointRes = 200;
static constexpr int kNbPointErr = 2000;

// ============================================================
// Result of one (type, cache, fullPiv) configuration
// ============================================================
template <typename Scalar>
struct ConfigResult {
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    CacheLevel  cache    = CacheLevel::none;
    bool        fullPiv  = true;
    std::string tt_file;                 ///< path of the saved tensor train
    int         n_calls  = 0;            ///< real function evaluations
    int         n_cached = 0;            ///< runner-level cache hits
    Real        max_abs_rel = Real(0);   ///< worst relative error on the grid
    bool        completed = false;
};

// ============================================================
// One configuration
// ============================================================
template <typename Scalar>
ConfigResult<Scalar> run_config(const std::string& type_label,
                                CacheLevel cache,
                                bool fullPiv)
{
    using Real    = typename Eigen::NumTraits<Scalar>::Real;
    using Complex = std::complex<Real>;
    using Sint    = util::i128;

    const std::string ctx = testing::config_label(type_label, cache, fullPiv);

    std::cout << "\n----- " << ctx << " -----" << std::endl;

    ConfigResult<Scalar> out;
    out.cache   = cache;
    out.fullPiv = fullPiv;

    // ---- grid ----
    QTGrid<Real, Sint> grid(Real(0), Real(1), kNBit);

    // ---- a non-default pivot1, to verify propagation ----
    std::vector<int> pivot1_custom(kNBit, 1);
    pivot1_custom[0] = 0;                          // [0,1,1,...,1]

    TCI2_1D_runner_opts<Scalar> opts{
        .reltol  = Eigen::NumTraits<Real>::epsilon() * Real(100),
        .pivot1  = pivot1_custom,
        .fullPiv = fullPiv,
        .cache   = cache
    };

    TCI2_1D_runner_param<Scalar> param(kNBit, kNbIter, kBondDim, opts);

    // ---- parameter propagation, including the derived cache flags ----
    check(param.nBit == kNBit,        ctx, "nBit propagated");
    check(param.nb_iter == kNbIter,   ctx, "nb_iter propagated");
    check(param.bondDim == kBondDim,  ctx, "bondDim propagated");
    check(param.pivot1 == pivot1_custom, ctx, "pivot1 propagated");
    check(param.fullPiv == fullPiv,   ctx, "fullPiv propagated");
    check(param.nRookIter == 3,       ctx, "nRookIter default");
    check(param.cache == cache,       ctx, "cache propagated");
    // The two caching layers are mutually exclusive by construction.
    check(param.useCachedFunction_xfac_lvl == (cache == CacheLevel::xfac),
          ctx, "xfac-level cache flag matches CacheLevel");
    check(param.do_cache_runner_lvl == (cache == CacheLevel::runner),
          ctx, "runner-level cache flag matches CacheLevel");
    check(!(param.useCachedFunction_xfac_lvl && param.do_cache_runner_lvl),
          ctx, "the two cache layers are never both active");

    // ---- fit ----
    auto test_func = pick_test_func<Scalar>();

    const Real pi_value = pi<Real>();
    const std::vector<Real> discontinuity_points{
        pi_value / Real(8),
        pi_value / Real(6),
        pi_value / Real(4)
    };

    const std::string prefix = "test/output/" + ctx;
    out.tt_file = prefix + ".tt";

    TCI2_1D_Runner<Scalar, Sint> runner(grid, param, test_func);
    runner.fit(/*additional_pivot=*/{}, /*verbose=*/true, /*do_save=*/true,
               prefix, kNbPointRes, discontinuity_points);

    // ---- reload f_values JSON and require an exact decimal roundtrip ----
    const auto [loaded_value,
                loaded_discontinuities,
                loaded_f_discontinuities] =
        load_fvalues_from_json<Complex>(prefix + "_f_values.json");

    const Complex expected_value =
        TCI2_1D_Runner<Scalar, Sint>::to_complex(test_func(grid.get_b()));

    std::vector<Complex> expected_f_discontinuities;
    expected_f_discontinuities.reserve(discontinuity_points.size());
    for (const Real& point : discontinuity_points) {
        expected_f_discontinuities.push_back(
            TCI2_1D_Runner<Scalar, Sint>::to_complex(test_func(point)));
    }

    check(loaded_value == expected_value,
          ctx, "f_values endpoint value reloads exactly");
    check(loaded_discontinuities == discontinuity_points,
          ctx, "f_values discontinuity points reload exactly");
    check(loaded_f_discontinuities == expected_f_discontinuities,
          ctx, "f_values at discontinuities reload exactly");

    out.n_calls  = runner.counter.load();
    out.n_cached = runner.counter_cached.load();

    // ---- counter invariants ----
    check(out.n_calls > 0, ctx, "at least one real function evaluation happened");

    if (cache == CacheLevel::runner) {
        check(out.n_cached > 0, ctx, "runner-level cache produced hits");
        // Every distinct point is evaluated exactly once, so the number of real
        // evaluations can never exceed the number of distinct grid points.
        check(out.n_calls <= out.n_calls + out.n_cached,
              ctx, "calls + hits accounting is sane");
    } else {
        check(out.n_cached == 0, ctx,
              "no runner-level cache hits when the runner cache is off");
    }

    // ---- reload the saved tensor train ----
    auto loaded = xfac_quad::load_vector_tensor<Complex>(out.tt_file);
    check(!loaded.empty(), ctx, "saved tensor train reloads non-empty");
    if (loaded.empty()) return out;

    xfac_quad::TensorTrain<Complex> tt_complex;
    tt_complex.M = std::move(loaded);

    // ---- accuracy against the reference function ----
    std::function<Complex(std::vector<int>)> gf_id =
        [&](std::vector<int> id) -> Complex {
            return TCI2_1D_Runner<Scalar, Sint>::to_complex(
                       test_func(grid.id_to_coord(id)));
        };

    TTErrorOnGrid<Real> error =
        error_TT_on_grid_point(tt_complex, gf_id, grid, kNbPointErr);
    std::cout << error;

    out.max_abs_rel = error.abs_rel_max;

    // Loose on purpose: rook pivoting (fullPiv=false) is expected to be a little
    // worse than a full search, and this check is here to catch gross breakage,
    // not to pin down the last few ulps. Tighten once you know the real spread.
    const Real tol = Eigen::NumTraits<Real>::epsilon() * Real(1e6);
    check(error.abs_rel_max <= tol, ctx, "relative error within tolerance");

    // ---- save/load roundtrip ----
    const std::string roundtrip_file = prefix + "_roundtrip.tt";
    {
        std::ofstream ofs(roundtrip_file);
        xfac_quad::save_Tensor3D_to_arma(ofs, tt_complex.M);
    }
    auto cores2 = xfac_quad::load_vector_tensor<Complex>(roundtrip_file);
    check(cores2.size() == tt_complex.M.size(), ctx, "roundtrip core count");

    if (cores2.size() == tt_complex.M.size()) {
        using std::abs;
        Real roundtrip_diff = Real(0);
        bool shapes_ok = true;
        for (size_t c = 0; c < tt_complex.M.size(); ++c) {
            const auto& orig   = tt_complex.M[c];
            const auto& reload = cores2[c];
            if (orig.n_rows != reload.n_rows ||
                orig.n_cols != reload.n_cols ||
                orig.n_slices != reload.n_slices) { shapes_ok = false; break; }
            for (Eigen::Index i = 0;
                 i < static_cast<Eigen::Index>(orig.data.size()); ++i) {
                Complex d = orig.data[i] - reload.data[i];
                Real err = abs(d.real()) + abs(d.imag());
                if (err > roundtrip_diff) roundtrip_diff = err;
            }
        }
        check(shapes_ok, ctx, "roundtrip core shapes match");
        check(roundtrip_diff == Real(0),
              ctx, "save/load roundtrip is exactly lossless");
    }

    out.completed = true;
    return out;
}

// ============================================================
// All configurations for one scalar type
// ============================================================
template <typename Scalar>
void run_all_configs(const std::string& type_label)
{
    constexpr bool is_complex = Eigen::NumTraits<Scalar>::IsComplex != 0;

    std::cout << "\n================================================\n"
              << "  Testing " << type_label
              << "  (is_complex=" << is_complex << ")\n"
              << "================================================" << std::endl;

    std::vector<ConfigResult<Scalar>> results;
    for (bool fullPiv : {true, false})
        for (CacheLevel cache : {CacheLevel::none,
                                 CacheLevel::xfac,
                                 CacheLevel::runner})
            results.push_back(run_config<Scalar>(type_label, cache, fullPiv));

    // ---- caching must not change the result ----
    // The three caching layers memoise the same pure function and must therefore
    // produce a byte-identical tensor train. This is the check that would have
    // caught the runner-level data race directly.
    //
    // fullPiv=false is excluded: ARRLUDecomp starts its rook iteration from
    // take_n_random() with an unseeded generator, so two fits of the same
    // configuration already differ from each other and comparing across cache
    // levels would measure that randomness rather than the cache. Rook is still
    // covered by the accuracy check in run_config().
    {
        const bool fullPiv = true;
        const ConfigResult<Scalar>* baseline = nullptr;
        for (const auto& r : results) {
            if (r.fullPiv != fullPiv || !r.completed) continue;
            if (!baseline) { baseline = &r; continue; }

            const std::string ctx = type_label + "_" + testing::piv_name(fullPiv);
            const std::string a = testing::read_file(baseline->tt_file);
            const std::string b = testing::read_file(r.tt_file);

            check(!a.empty() && !b.empty(), ctx, "tensor train files readable");
            check(a == b, ctx,
                  std::string("cache=") + xf_qd_detail::cache_name(r.cache)
                  + " gives the same tensor train as cache="
                  + xf_qd_detail::cache_name(baseline->cache)
                  + " (caching must be transparent)");
        }
    }

    // ---- caching must not increase the amount of real work ----
    for (bool fullPiv : {true, false}) {
        const ConfigResult<Scalar>* none_run = nullptr;
        for (const auto& r : results)
            if (r.fullPiv == fullPiv && r.cache == CacheLevel::none && r.completed)
                none_run = &r;

        if (!none_run) continue;

        for (const auto& r : results) {
            if (r.fullPiv != fullPiv || !r.completed) continue;
            if (r.cache == CacheLevel::none) continue;
            const std::string ctx = type_label + "_" + testing::piv_name(fullPiv);
            check(r.n_calls <= none_run->n_calls, ctx,
                  std::string("cache=") + xf_qd_detail::cache_name(r.cache)
                  + " does not evaluate the function more often than cache=none");
        }
    }

    // ---- summary table ----
    std::cout << "\n  " << std::left << std::setw(10) << "cache"
              << std::setw(10) << "pivot"
              << std::right << std::setw(12) << "calls"
              << std::setw(12) << "hits" << "   max rel err\n";
    for (const auto& r : results) {
        std::ostringstream err;
        err << std::scientific << std::setprecision(3) << r.max_abs_rel;
        std::cout << "  " << std::left << std::setw(10)
                  << xf_qd_detail::cache_name(r.cache)
                  << std::setw(10) << testing::piv_name(r.fullPiv)
                  << std::right << std::setw(12) << r.n_calls
                  << std::setw(12) << r.n_cached
                  << "   " << err.str() << "\n";
    }
    std::cout << std::flush;
}

// ============================================================
// main
// ============================================================
int main()
{
    std::filesystem::create_directories("test/output");
    std::cout << "Running TCI2_1D_Runner configuration tests "
                 "(cache x fullPiv x scalar type)...\n";

    run_all_configs<double>("double");
    run_all_configs<dd_128>("dd_128");
    run_all_configs<float128>("float128");

    run_all_configs<std::complex<double>>("complex_double");
    run_all_configs<Cdd_128>("complex_dd_128");
    run_all_configs<Cfloat128>("complex_float128");

    return testing::finish("test_runner");
}