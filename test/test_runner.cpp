#include "xf-qd-runner.hpp"
#include "xf-quad-param.hpp"

#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>
#include <complex>
#include <algorithm>
#include <filesystem>

// ============================================================
// Templated test functions — take Real, return Scalar
// ============================================================

/// f1: real-valued — product and sum of cos
template <typename T>
T f1_real(T x) {
    using std::cos;
    return cos(x) + cos(T(2)*x) * cos(T(3)*x)
                  + cos(T(4)*x) * cos(T(5)*x);
}

/// f2: complex-valued — mix of cos (real) and sin (imag)
template <typename T>
std::complex<T> f2_complex(T x) {
    using std::cos;
    using std::sin;
    T re = cos(x) * cos(T(2)*x) + cos(T(3)*x);
    T im = sin(x) * cos(T(2)*x) + sin(T(3)*x) * sin(T(4)*x);
    return std::complex<T>(re, im);
}

// ============================================================
// Helper: convert integer k to bit vector of length nBit
// ============================================================
template <typename Sint>
std::vector<int> k_to_id(Sint k, int nBit) {
    std::vector<int> id(nBit, 0);
    for (int j = 0; j < nBit; ++j)
        id[j] = static_cast<int>((k >> j) & Sint(1));
    return id;
}

// ============================================================
// Helper: max abs diff between two TensorTrains
// ============================================================
template <typename Scalar>
auto max_error_between_tt(
    const xfac_quad::TensorTrain<Scalar>& tt1,
    const xfac_quad::TensorTrain<Scalar>& tt2,
    int nBit, int nSamples) -> typename Eigen::NumTraits<Scalar>::Real
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using std::abs;
    Real max_err = Real(0);
    util::i128 N = util::i128(1) << nBit;
    for (int i = 0; i < nSamples; ++i) {
        util::i128 k = (util::i128(i) * util::i128(1234567) + util::i128(7654321)) % N;
        auto id = k_to_id(k, nBit);
        Scalar v1 = tt1.eval(id);
        Scalar v2 = tt2.eval(id);
        Real err = [&]() {
            using std::abs;
            return abs(v1 - v2);
        }();
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ============================================================
// Test runner for one Scalar type
// ============================================================
template <typename Scalar>
void run_test(const std::string& label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Complex = std::complex<Real>;
    using Sint = util::i128;

    constexpr bool is_complex = Eigen::NumTraits<Scalar>::IsComplex != 0;

    std::cout << "\n===== Testing " << label
              << " (is_complex=" << is_complex << ") =====\n" << std::endl;

    // ---- grid (always on Real) ----
    const Real a = Real(0);
    const Real b = Real(1);
    const int nBit = 30;
    QTGrid<Real, Sint> grid(a, b, nBit);

    // ---- build a non-default pivot1 to verify propagation ----
    std::vector<int> pivot1_custom(nBit, 1);
    pivot1_custom[0] = 0;              // [0,1,1,...,1]

    // ---- params: set every available option non-default ----
    TCI2_1D_runner_opts<Scalar> opts{
        .reltol    = Eigen::NumTraits<Real>::epsilon() * Real(100),
        .pivot1    = pivot1_custom,
        .fullPiv   = true,
        .cache     = CacheLevel::runner
    };

    TCI2_1D_runner_param<Scalar> param(nBit, 10, 100, opts);

    std::cout << "Parameters:\n" << param.to_string() << "\n" << std::endl;

    assert(param.nBit == nBit);
    assert(param.nb_iter == 10);
    assert(param.bondDim == 100);
    assert(param.pivot1 == pivot1_custom);
    assert(param.fullPiv == true);
    assert(param.nRookIter == 3);   // default
    assert(param.cache == CacheLevel::runner);
    assert(param.useCachedFunction_xfac_lvl == false);
    assert(param.do_cache_runner_lvl == true);
    std::cout << "[OK] All parameter fields propagated correctly.\n";

    // ---- pick the right test function ----
    std::function<Scalar(Real)> test_func;
    if constexpr (is_complex) {
        test_func = f2_complex<Real>;
    } else {
        test_func = f1_real<Real>;
    }
    std::cout << "Testing with " << (is_complex ? "f2_complex" : "f1_real") << "\n";

    // ---- fit ----
    const std::string prefix = "test/output/test_" + label;
    std::vector<Real> additional_pivot = {};

    TCI2_1D_Runner<Scalar, Sint> runner(grid, param, test_func);
    runner.fit(additional_pivot, /*verbose=*/true, /*do_save=*/true,
               prefix, /*nb_point_res=*/200);

    // ---- load saved TT ----
    std::string tt_file = prefix + ".tt";
    auto loaded_complex = xfac_quad::load_vector_tensor<Complex>(tt_file);
    assert(!loaded_complex.empty());

    // ---- error using error_TT_on_grid_point ----
    xfac_quad::TensorTrain<Complex> tt_complex;
    tt_complex.M = std::move(loaded_complex);

    std::function<Complex(std::vector<int>)> gf_id = [&](std::vector<int> id) -> Complex {
        Real x = grid.id_to_coord(id);
        return TCI2_1D_Runner<Scalar, Sint>::to_complex(test_func(x));
    };

    TTErrorOnGrid<Real> error =
        error_TT_on_grid_point(tt_complex, gf_id, grid, 2000);
    std::cout << error;

    // ---- save/load roundtrip: compare cores element-by-element ----
    std::string roundtrip_file = prefix + "_roundtrip.tt";
    {
        std::ofstream ofs(roundtrip_file);
        xfac_quad::save_Tensor3D_to_arma(ofs, tt_complex.M);
    }
    auto cores2 = xfac_quad::load_vector_tensor<Complex>(roundtrip_file);
    assert(cores2.size() == tt_complex.M.size());

    Real roundtrip_diff = Real(0);
    using std::abs;
    for (size_t c = 0; c < tt_complex.M.size(); ++c) {
        const auto& orig = tt_complex.M[c];
        const auto& reload = cores2[c];
        assert(orig.n_rows == reload.n_rows);
        assert(orig.n_cols == reload.n_cols);
        assert(orig.n_slices == reload.n_slices);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(orig.data.size()); ++i) {
            Complex diff = orig.data[i] - reload.data[i];
            Real err = abs(diff.real()) + abs(diff.imag());
            if (err > roundtrip_diff) roundtrip_diff = err;
        }
    }
    std::cout << "Max element-wise error between save/load: " << roundtrip_diff << "\n";
    assert(roundtrip_diff <= Eigen::NumTraits<Real>::epsilon() * Real(10));
    std::cout << "[OK] Save/load roundtrip is lossless.\n";

    std::cout << "\n===== " << label << " PASSED =====\n" << std::endl;
}

// ============================================================
// main
// ============================================================
int main() {
    std::filesystem::create_directories("test/output");
    std::cout << "Running TCI2_1D_Runner tests...\n";

    // real Scalar -> f1_real
    run_test<double>("double");
    run_test<dd_128>("dd_128");
    run_test<float128>("float128");

    // complex Scalar -> f2_complex
    run_test<std::complex<double>>("complex_double");
    run_test<Cdd_128>("complex_dd_128");
    run_test<Cfloat128>("complex_float128");

    std::cout << "\nAll tests passed.\n";
    return 0;
}
