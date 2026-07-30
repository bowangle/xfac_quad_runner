#pragma once

#include <vector>
#include <iostream>
#include <complex>
#include <functional>
#include <fstream>
#include <string>

#include <grid.h>
#include <xfac_quad/xfac_quad.hpp>

template <typename Scalar>
void compute_stats(
    const std::vector<Scalar>& v,
    Scalar& mean,
    Scalar& min,
    Scalar& max)
{
    using Real = Scalar;
    using std::abs;
    using boost::multiprecision::abs;

    Real sum = Real(0);

    Real min_abs = std::numeric_limits<Real>::infinity();
    Real max_abs = Real(0);

    int count = 0;

    for (const auto& x : v)
    {
        Real ax = abs(x);

        sum += ax;

        if (ax < min_abs) min_abs = ax;
        if (ax > max_abs) max_abs = ax;

        count++;
    }

    Real mean_abs = (count > 0)
        ? sum / Real(count)
        : std::numeric_limits<Real>::quiet_NaN();

    // cast back to Scalar (important if Scalar is complex)
    mean = Scalar(mean_abs);
    min  = Scalar(min_abs);
    max  = Scalar(max_abs);
}

// output type of error calculation
template <typename Scalar>
struct TTErrorOnGrid {
    using Complex = std::complex<Scalar>;

    std::vector<Scalar> real_abs;
    std::vector<Scalar> real_rel;
    std::vector<Scalar> imag_abs;
    std::vector<Scalar> imag_rel;
    std::vector<Scalar> abs_abs;
    std::vector<Scalar> abs_rel;

    std::vector<Scalar> l_point;
    std::vector<Complex> l_ref;
    std::vector<Complex> l_tt;

    Scalar real_abs_mean, real_abs_min, real_abs_max;
    Scalar real_rel_mean, real_rel_min, real_rel_max;

    Scalar imag_abs_mean, imag_abs_min, imag_abs_max;
    Scalar imag_rel_mean, imag_rel_min, imag_rel_max;

    Scalar abs_abs_mean, abs_abs_min, abs_abs_max;
    Scalar abs_rel_mean, abs_rel_min, abs_rel_max;

    void compute_summary()
    {
        compute_stats(real_abs, real_abs_mean, real_abs_min, real_abs_max);
        compute_stats(real_rel, real_rel_mean, real_rel_min, real_rel_max);

        compute_stats(imag_abs, imag_abs_mean, imag_abs_min, imag_abs_max);
        compute_stats(imag_rel, imag_rel_mean, imag_rel_min, imag_rel_max);

        compute_stats(abs_abs, abs_abs_mean, abs_abs_min, abs_abs_max);
        compute_stats(abs_rel, abs_rel_mean, abs_rel_min, abs_rel_max);
    }
};

template <typename Scalar>
void save_TTErrorOnGrid(const TTErrorOnGrid<Scalar>& e,
                        const std::string& filename)
{
    std::ofstream file(filename);

    if (!file)
        throw std::runtime_error("Cannot open file");

    file << std::setprecision(std::numeric_limits<Scalar>::max_digits10);

    using Complex = std::complex<Scalar>;

    auto write_vec = [&](const char* name, const std::vector<Scalar>& v)
    {
        file << name << " " << v.size() << "\n";
        for (const auto& x : v) file << x << " ";
        file << "\n\n";
    };

    auto write_cvec = [&](const char* name, const std::vector<Complex>& v)
    {
        file << name << " " << v.size() << "\n";
        for (const auto& x : v)
            file << "(" << x.real() << "," << x.imag() << ") ";
        file << "\n\n";
    };

    write_vec("l_point",  e.l_point);
    write_cvec("l_ref", e.l_ref);
    write_cvec("l_tt",  e.l_tt);
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const TTErrorOnGrid<Scalar>& e)
{
    auto print_stats = [&](const char* name,
                           Scalar mean,
                           Scalar min,
                           Scalar max)
    {
        os << name << ":\n"
           << "  mean = " << mean << "\n"
           << "  min  = " << min << "\n"
           << "  max  = " << max << "\n";
    };

    os << "================ TT ERROR REPORT ================\n";

    print_stats("Real abs error", e.real_abs_mean, e.real_abs_min, e.real_abs_max);
    print_stats("Real rel error", e.real_rel_mean, e.real_rel_min, e.real_rel_max);

    print_stats("Imag abs error", e.imag_abs_mean, e.imag_abs_min, e.imag_abs_max);
    print_stats("Imag rel error", e.imag_rel_mean, e.imag_rel_min, e.imag_rel_max);

    print_stats("Abs error", e.abs_abs_mean, e.abs_abs_min, e.abs_abs_max);
    print_stats("Abs rel error", e.abs_rel_mean, e.abs_rel_min, e.abs_rel_max);

    os << "=================================================\n";

    return os;
}

// Error calculation:
template <typename Scalar, typename Sint>
TTErrorOnGrid<Scalar> error_TT_on_grid_point(
    const TensorTrain<std::complex<Scalar>>& tt,
    std::function<std::complex<Scalar>(MultiIndex)> gf_id,
    const QTGrid<Scalar, Sint>& grid,
    int nb_point)
{
    using Complex = std::complex<Scalar>;

    TTErrorOnGrid<Scalar> out;
    out.real_abs.resize(nb_point);
    out.real_rel.resize(nb_point);
    out.imag_abs.resize(nb_point);
    out.imag_rel.resize(nb_point);
    out.abs_abs.resize(nb_point);
    out.abs_rel.resize(nb_point);

    std::vector<Scalar> l_point = linspace(grid.get_a(), grid.get_b(), nb_point);
    std::vector<Complex> l_ref_i(nb_point);
    std::vector<Complex> l_res_tt(nb_point);


    for (int i = 0; i < nb_point; i++)
    {
        Scalar x = l_point[i];
        const MultiIndex id = grid.coord_to_id(x);

        Complex ref_i = gf_id(id);
        l_ref_i[i] = ref_i;

        Complex res_tt = tt.eval(id.data);
        l_res_tt[i] = res_tt;

        Complex diff = ref_i - res_tt;

        Scalar abs_diff = std::abs(diff);
        Scalar ref_norm = std::abs(ref_i);

        // signed errors
        out.real_abs[i] = diff.real();
        out.imag_abs[i] = diff.imag();
        out.abs_abs[i]  = abs_diff;

        // relative
        if (ref_norm != Scalar(0))
        {
            out.real_rel[i] = diff.real() / ref_norm;
            out.imag_rel[i] = diff.imag() / ref_norm;
            out.abs_rel[i]  = abs_diff / ref_norm;
        }
        else
        {
            out.real_rel[i] = 0;
            out.imag_rel[i] = 0;
            out.abs_rel[i]  = 0;
        }
    }
    out.l_point = l_point;
    out.l_ref = l_ref_i;
    out.l_tt = l_res_tt;
    out.compute_summary();
    return out;
}