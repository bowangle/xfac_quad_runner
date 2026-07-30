#pragma once

// Shared helpers for the xf_qd_runner tests.
//
// NOTE ON assert(): the standard assert() is compiled out whenever NDEBUG is
// defined, which is the default for CMake's Release and RelWithDebInfo builds.
// compile.sh happens to override CMAKE_CXX_FLAGS_RELEASE and so drops -DNDEBUG,
// but any plain `cmake -DCMAKE_BUILD_TYPE=Release` would silently turn every
// assert() into a no-op. The check() below always runs.

#include "xf-qd-runner.hpp"
#include "xf-quad-param.hpp"

#include <complex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// Failure accounting (never compiled out)
// ============================================================
namespace testing {

inline int& failure_count() { static int n = 0; return n; }

inline void check(bool ok, const std::string& ctx, const std::string& what)
{
    if (!ok) {
        std::cerr << "[FAIL] " << ctx << " :: " << what << std::endl;
        ++failure_count();
    }
}

/// Report and turn the accumulated failures into a process exit code.
inline int finish(const std::string& suite)
{
    if (failure_count() == 0) {
        std::cout << "\n[PASS] " << suite << ": all checks passed.\n";
        return 0;
    }
    std::cerr << "\n[FAIL] " << suite << ": " << failure_count()
              << " check(s) failed.\n";
    return 1;
}

/// Read a whole file as raw bytes; empty string if it cannot be opened.
inline std::string read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// Index of the first line that differs, or -1 if the texts are identical.
/// Used to make log diffs readable instead of dumping thousands of characters.
inline int first_differing_line(const std::string& a, const std::string& b,
                                std::string& line_a, std::string& line_b)
{
    std::istringstream sa(a), sb(b);
    std::string la, lb;
    int n = 0;
    while (true) {
        bool oka = static_cast<bool>(std::getline(sa, la));
        bool okb = static_cast<bool>(std::getline(sb, lb));
        if (!oka && !okb) return -1;
        if (oka != okb || la != lb) {
            line_a = oka ? la : "<end of output>";
            line_b = okb ? lb : "<end of output>";
            return n;
        }
        ++n;
    }
}

/// Redirects std::cout for its lifetime so a fit() log can be captured and
/// compared. fit() writes through std::cout when verbose=true, so this gives
/// access to every sweep's pivot error at full precision without changing the
/// runner's interface.
class CoutCapture {
public:
    CoutCapture() : old_(std::cout.rdbuf(buffer_.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(old_); }

    CoutCapture(const CoutCapture&)            = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;

    std::string str() const { return buffer_.str(); }

private:
    std::ostringstream buffer_;
    std::streambuf*    old_;
};

inline std::string piv_name(bool fullPiv) { return fullPiv ? "fullpiv" : "rook"; }

inline std::string config_label(const std::string& type_label,
                                CacheLevel cache,
                                bool fullPiv)
{
    return type_label + "_" + xf_qd_detail::cache_name(cache)
                      + "_" + piv_name(fullPiv);
}

} // namespace testing

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

/// Pick the test function matching the scalar type.
template <typename Scalar>
std::function<Scalar(typename Eigen::NumTraits<Scalar>::Real)> pick_test_func()
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex != 0)
        return f2_complex<Real>;
    else
        return f1_real<Real>;
}