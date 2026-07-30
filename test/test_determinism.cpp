// Determinism test.
//
// With full pivoting, TCI2 is a deterministic algorithm over a pure function, so
// two fits with identical inputs must agree bit for bit: same pivot error at
// every sweep, same call counts, same tensor train. Before the runner-level
// cache was made thread-safe this was false — concurrent inserts into one
// unordered_map caused lost values, which sent the pivot search down a different
// path and changed the numbers from run to run. That drift was visible in the
// logs long before it crashed, so it is worth asserting directly.
//
// Only fullPiv=true is covered here. With fullPiv=false, ARRLUDecomp seeds its
// rook iteration with take_n_random() from an unseeded generator, so repeat fits
// legitimately differ; test_runner still checks that rook fits are accurate.
//
// The comparison uses fit()'s own log, captured from std::cout. It contains
// every sweep's pivot error at max_digits10 plus the full error report, which
// makes it a far more sensitive fingerprint than the final numbers alone.

#include "test_common.hpp"

#include <filesystem>
#include <omp.h>

using testing::check;

// Same problem size as test_runner, so a determinism failure here corresponds
// directly to a configuration exercised there. Note this file runs every
// configuration twice.
static constexpr int kNBit    = 30;
static constexpr int kNbIter  = 10;
static constexpr int kBondDim = 100;
static constexpr int kNbPointRes = 200;

// ============================================================
// Everything one fit produced that ought to be reproducible
// ============================================================
struct RunSnapshot {
    std::string log;        ///< fit()'s verbose output, full precision
    std::string tt_bytes;   ///< the serialised tensor train
    int n_calls  = 0;
    int n_cached = 0;
};

template <typename Scalar>
RunSnapshot do_one_fit(CacheLevel cache, bool fullPiv, const std::string& prefix)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Sint = util::i128;

    QTGrid<Real, Sint> grid(Real(0), Real(1), kNBit);

    std::vector<int> pivot1_custom(kNBit, 1);
    pivot1_custom[0] = 0;

    TCI2_1D_runner_opts<Scalar> opts{
        .reltol  = Eigen::NumTraits<Real>::epsilon() * Real(100),
        .pivot1  = pivot1_custom,
        .fullPiv = fullPiv,
        .cache   = cache
    };
    TCI2_1D_runner_param<Scalar> param(kNBit, kNbIter, kBondDim, opts);

    auto test_func = pick_test_func<Scalar>();

    RunSnapshot snap;
    {
        // fit() logs through std::cout; capture it for the whole call.
        testing::CoutCapture capture;
        TCI2_1D_Runner<Scalar, Sint> runner(grid, param, test_func);
        runner.fit(/*additional_pivot=*/{}, /*verbose=*/true, /*do_save=*/false,
                   prefix, kNbPointRes);
        snap.n_calls  = runner.counter.load();
        snap.n_cached = runner.counter_cached.load();
        snap.log      = capture.str();
    }
    // fit() writes <prefix>.tt unconditionally, so it is available even with
    // do_save=false. Both runs use the same prefix, which keeps the logs
    // comparable; the bytes are read out before the next run overwrites them.
    snap.tt_bytes = testing::read_file(prefix + ".tt");
    return snap;
}

// ============================================================
// Compare two snapshots
// ============================================================
static void compare(const RunSnapshot& a, const RunSnapshot& b,
                    const std::string& ctx)
{
    check(a.n_calls == b.n_calls, ctx,
          "same number of function evaluations (" + std::to_string(a.n_calls)
          + " vs " + std::to_string(b.n_calls) + ")");
    check(a.n_cached == b.n_cached, ctx,
          "same number of cache hits (" + std::to_string(a.n_cached)
          + " vs " + std::to_string(b.n_cached) + ")");

    check(!a.tt_bytes.empty(), ctx, "tensor train was written");
    check(a.tt_bytes == b.tt_bytes, ctx, "identical tensor train");

    if (a.log != b.log) {
        std::string la, lb;
        const int line = testing::first_differing_line(a.log, b.log, la, lb);
        check(false, ctx,
              "fit logs diverge at line " + std::to_string(line + 1)
              + "\n         run 1: " + la
              + "\n         run 2: " + lb);
    } else {
        check(true, ctx, "identical fit log");
    }
}

// ============================================================
// One scalar type, all configurations
// ============================================================
template <typename Scalar>
void check_type(const std::string& type_label)
{
    std::cout << "\n===== determinism: " << type_label << " =====" << std::endl;

    // Only fullPiv=true is checked. With fullPiv=false, ARRLUDecomp picks its
    // starting rows/columns with take_n_random() from an unseeded generator, so
    // rook pivoting is nondeterministic by construction and repeat fits are not
    // expected to agree. Full pivoting returns before that code path is reached.
    const bool fullPiv = true;

    for (CacheLevel cache : {CacheLevel::none,
                             CacheLevel::xfac,
                             CacheLevel::runner}) {
        const std::string ctx = testing::config_label(type_label, cache, fullPiv);
        const std::string prefix = "test/output/det_" + ctx;

        const RunSnapshot a = do_one_fit<Scalar>(cache, fullPiv, prefix);
        const RunSnapshot b = do_one_fit<Scalar>(cache, fullPiv, prefix);

        compare(a, b, ctx);
        std::cout << "  " << ctx
                  << "  calls=" << a.n_calls
                  << " hits="   << a.n_cached << std::endl;
    }
}

// ============================================================
// Informational: does the result depend on the thread count?
// ============================================================
// This is deliberately NOT a failure. A difference here does not necessarily
// indicate a bug in the runner: a parallel reduction inside xfac can sum in a
// different order at a different thread count, and floating-point addition is
// not associative. It is reported because it cleanly separates "our cache is
// racy" (which also breaks the same-thread-count checks above) from "xfac's
// reductions are order-dependent" (which does not).
template <typename Scalar>
void report_thread_invariance(const std::string& type_label)
{
    const int saved = omp_get_max_threads();
    if (saved <= 1) {
        std::cout << "  (only one thread available; skipping)" << std::endl;
        return;
    }

    const std::string prefix = "test/output/det_threads_" + type_label;

    omp_set_num_threads(saved);
    const RunSnapshot many = do_one_fit<Scalar>(CacheLevel::runner, true, prefix);

    omp_set_num_threads(1);
    const RunSnapshot one = do_one_fit<Scalar>(CacheLevel::runner, true, prefix);

    omp_set_num_threads(saved);

    const bool same = (many.log == one.log) && (many.tt_bytes == one.tt_bytes);
    std::cout << "  " << type_label << ": " << saved << " threads vs 1 thread -> "
              << (same ? "identical" : "DIFFERENT (see comment in source)")
              << std::endl;
}

// ============================================================
// main
// ============================================================
int main()
{
    std::filesystem::create_directories("test/output");
    std::cout << "Running determinism tests "
                 "(each configuration fitted twice)...\n";

    check_type<double>("double");
    check_type<dd_128>("dd_128");
    check_type<float128>("float128");

    check_type<std::complex<double>>("complex_double");
    check_type<Cdd_128>("complex_dd_128");
    check_type<Cfloat128>("complex_float128");

    std::cout << "\n----- thread-count invariance (informational) -----"
              << std::endl;
    report_thread_invariance<double>("double");
    report_thread_invariance<float128>("float128");
    report_thread_invariance<Cfloat128>("complex_float128");

    return testing::finish("test_determinism");
}