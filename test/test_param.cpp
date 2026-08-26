// Exact save/load roundtrip tests for TCI2_1D_runner_param.

#include "test_common.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <nlohmann/json.hpp>

using testing::check;

template<typename Real>
bool exactly_equal(const Real& lhs, const Real& rhs)
{
    if constexpr (std::is_same_v<Real, dd_128>) {
        return lhs.x[0] == rhs.x[0] && lhs.x[1] == rhs.x[1];
    } else {
        return lhs == rhs;
    }
}

template<typename Scalar>
void test_roundtrip(const std::string& type_label)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    const Real reltol = Eigen::NumTraits<Real>::epsilon() * Real(100);
    const std::string ctx = "TCI2_1D_runner_param<" + type_label + ">";
    const std::string path =
        "test/output_param/param_" + type_label + ".json";

    TCI2_1D_runner_opts<Scalar> opts{
        .reltol = reltol,
        .pivot1 = {1, 0, 1, 1, 0, 0, 1},
        .fullPiv = true,
        .nRookIter = 17,
        .cache = CacheLevel::runner
    };
    const TCI2_1D_runner_param<Scalar> saved(7, 123, 45, opts);

    saved.save(path);

    std::ifstream input(path);
    nlohmann::json saved_json;
    input >> saved_json;
    check(!saved_json.contains("pivot1"), ctx, "pivot1 is not saved");

    const TCI2_1D_runner_param<Scalar> loaded(path, saved.pivot1);
    const TCI2_1D_runner_param<Scalar> loaded_without_pivot(path);

    check(saved.nBit == loaded.nBit, ctx, "nBit roundtrips exactly");
    check(saved.nb_iter == loaded.nb_iter, ctx, "nb_iter roundtrips exactly");
    check(saved.bondDim == loaded.bondDim, ctx, "bondDim roundtrips exactly");
    check(exactly_equal(saved.reltol, loaded.reltol), ctx,
          "reltol roundtrips exactly");
    check(saved.pivot1 == loaded.pivot1, ctx,
          "pivot1 can be supplied when loading");
    check(loaded_without_pivot.pivot1.empty(), ctx,
          "pivot1 defaults to empty when loading");
    check(saved.fullPiv == loaded.fullPiv, ctx, "fullPiv roundtrips exactly");
    check(saved.nRookIter == loaded.nRookIter, ctx,
          "nRookIter roundtrips exactly");
    check(saved.cache == loaded.cache, ctx, "cache roundtrips exactly");
    check(saved.useCachedFunction_xfac_lvl ==
              loaded.useCachedFunction_xfac_lvl,
          ctx, "xfac cache flag roundtrips exactly");
    check(saved.do_cache_runner_lvl == loaded.do_cache_runner_lvl,
          ctx, "runner cache flag roundtrips exactly");
}

int main()
{
    std::filesystem::create_directories("test/output_param");

    test_roundtrip<double>("double");
    test_roundtrip<dd_128>("dd_128");
    test_roundtrip<float128>("float128");

    return testing::finish("TCI2_1D_runner_param save/load roundtrip");
}
