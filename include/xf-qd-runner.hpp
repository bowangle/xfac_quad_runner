#pragma once

#include <complex>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <unordered_map>
#include <memory>

#include <spdlog/spdlog.h>

#include <grid.h> // QTgrid_quad.
#include <xfac_quad/xfac_quad.hpp>

#include "xf-quad-param.hpp"

template <typename Scalar, typename Sint>
class TCI2_1D_Runner{
    public:
    
    using Complex = std::complex<Scalar>;

    const QTGrid<Scalar, Sint> grid;
    const TCI2_1D_runner_param<Scalar> tci_param;
    std::function<Complex(Scalar)> function_x;  // original function
    std::function<Complex(std::vector<int>)> function_id; // function on grid
    const std::vector<int> l_d;
    int counter = 0;
    int counter_cached = 0;
    std::shared_ptr<spdlog::logger> const logger;

    TCI_Runner(
        QTGrid<Scalar, Sint> grid_,
        TCI2_1D_runner_param<Scalar> tci_param_,
        std::function<Complex(Scalar)> function_x_,
        std::shared_ptr<spdlog::logger> logger_ = nullptr
    )
    : 
        grid(grid_),
        tci_param(tci_param_),
        function_x(function_x_),
        function_id(resolve_function(tci_param.do_cache, function_x_, grid_, logger_)),
        l_d(std::vector<int>(grid_.get_nBits(), 2)),
        counter(0),         // count the number of uncached call. (if no cache, then all the call)
        counter_cached(0),   // count the total number of call (in absence of cached, it's 0)
        logger(logger_)
    {}

    std::function<Complex(std::vector<int>)>
    func_to_grid(
        std::function<Complex(Scalar)> f,
        const QTGrid<Scalar, Sint>& qt_grid, // be carefull of lifetime. The output function can't outlive the TCI_runner class. A full copy would be totaly free.
        std::shared_ptr<spdlog::logger> logger_ = nullptr
    )
    {
        if (logger_){
                return [this, &qt_grid, f](const std::vector<int>& id) -> Complex {
                    counter++;
                    return f(qt_grid.id_to_coord(id.data));
                };
            }
        else{
            return [&qt_grid, f](const std::vector<int>& id) -> Complex {
                    return f(qt_grid.id_to_coord(id.data));
                };
        }
    }

    struct MultiIndexHash {
        size_t operator()(std::vector<int> const& mi) const noexcept {
            size_t seed = mi.size();
            for (char32_t x : mi) {
                seed ^= static_cast<size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    std::function<Complex(std::vector<int>)>
    make_cached_function(std::function<Complex(std::vector<int>)> f)
    {
        auto cache = std::make_shared<std::unordered_map<std::vector<int>, Complex, MultiIndexHash>>();

        return [this, f = std::move(f), cache](const std::vector<int>& v) mutable -> Complex
        {
            this->counter_cached++;
            auto it = cache->find(v);
            if (it != cache->end())
                return it->second;

            Complex result = f(v);
            cache->emplace(v, result);

            return result;
        };
    }

    std::function<Complex(std::vector<int>)>
    resolve_function(
        bool do_cache,
        std::function<Complex(Scalar)> function_x_,
        const QTGrid<Scalar, Sint>& grid_, // be carefull of lifetime. The output function can't outlive the TCI_runner class. A full copy would be totaly free.
        std::shared_ptr<spdlog::logger> logger_ = nullptr
    ){
        std::function<Complex(std::vector<int>)> function_on_grid = func_to_grid(function_x_, grid_, logger_);
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

    std::string to_string_for_save(const std::vector<Complex>& v)
    {
        std::ostringstream oss;
        oss << "[";

        for (size_t i = 0; i < v.size(); ++i)
        {
            oss << "\"(" << to_string(v[i]) << ")\"";
            if (i + 1 < v.size())
                oss << ", ";
        }

        oss << "]";
        return oss.str();
    }

    std::string to_string(const Complex& z)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<Scalar>::digits10 + 5) << std::scientific;
        oss << z.real();
        // always print sign explicitly for imaginary part
        if (z.imag() >= 0)
            oss << "+";
        oss << z.imag() << "j";
        return oss.str();
    }

    std::string to_string(const std::vector<Scalar>& v)
    {
        std::ostringstream oss;

        oss << std::setprecision(std::numeric_limits<Scalar>::digits10 + 5) << std::scientific;

        oss << "[";

        for (size_t i = 0; i < v.size(); ++i)
        {
            oss << v[i];

            if (i + 1 < v.size())
                oss << ", ";
        }

        oss << "]";

        return oss.str();
    }

    void save_to_json(
        const std::string& filename,
        const Complex& value,
        const std::vector<Scalar>& l_discontinuity,
        const std::vector<Complex>& l_f_discontinuity)
    {
        // this hard implement the json file, easier to do it that way here
        std::ofstream file(filename);

        file << "{\n";

        file << "  \"value\": \"" << to_string(value) << "\",\n";

        file << "  \"l_discontinuity\": "
            << to_string(l_discontinuity) << ",\n";

        file << "  \"l_f_discontinuity\": "
            << to_string_for_save(l_f_discontinuity) << "\n";

        file << "}\n";
    }

    void fit(
        const std::vector<Scalar> additional_pivot,
        bool verbose = true,
        bool do_save = false,
        const std::string& file_prefix="",
        int nb_point_res=1000,
        const std::vector<Scalar> E_discontinuity={},
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

        // initial pivot
        const Scalar E_init = grid.id_to_coord(tci_param.pivot1);

        // additional pivot
        std::vector<std::vector<int>> id_additionnal_pivot;
        id_additionnal_pivot.reserve(additional_pivot.size());
        for (const auto& E : other_E)
            id_additionnal_pivot.push_back(grid.coord_to_id(E));

        // ===================== printing parameter ======================
        {
            log("========== FIT PARAMETERS ==========");

            log("E_init = " + to_string_stream(E_init));
            log("E_init id = " + to_string_stream(pivot1));

            std::string other_E_str = "[";
            for (size_t i = 0; i < other_E.size(); ++i) {
                other_E_str += to_string_stream(other_E[i]);
                if (i + 1 < other_E.size()) other_E_str += ", ";
            }
            other_E_str += "]";

            log("additional_pivot " + additional_pivot);
            log("id_additionnal_pivot " + id_additionnal_pivot);

            log("verbose = " + to_string_stream(verbose ? "true" : "false"));
            log("do_save = " + to_string_stream(do_save ? "true" : "false"));

            log("file_prefix = " + file_prefix);

            log("nb_point_res = " + to_string_stream(nb_point_res));

            log(tci_param.to_string());
        }
        // ===================== optional parameter ======================

        if (!E_discontinuity.empty()){
            // if not the case we skip the f_value file generation
            Complex f_a = function_x(grid.get_a());
            Complex f_b = function_x(grid.get_b());

            std::vector<Complex> l_f_discontinuity;
            l_f_discontinuity.reserve(E_discontinuity.size());

            for (const Scalar& e : E_discontinuity) {
                l_f_discontinuity.push_back(function_x(e));
            }
            
            log("  f_a = " + to_string(f_a));
            log("  f_b = " + to_string(f_b));
            log("  E_discontinuity = " + to_string(E_discontinuity));
            log("  l_f_discontinuity = " + to_string_for_save(l_f_discontinuity));
            save_to_json(
                file_prefix + "_f_values.json",
                f_b,
                E_discontinuity,
                l_f_discontinuity
            );
        }

        // ===================== actual computation ======================

        log("==========begin iteration===============");

        std::vector<int> local_dim(nBit, 2);    // each bit can take two value 0 or 1

        const xfac_quad::TensorFunction<Complex> TF_function xfac_quad::TensorFunction<Complex>(function_id, TCI2_1D_runner_param.useCachedFunction_xfac_lvl);
        xfac_quad::TensorCI2<Complex> tci xfac_quad::TensorCI2(/*TensorFunction<T>=*/ TF_function, /*vector<int>=*/ localDim, /*TensorCI2Param<T>=*/ TCI2_1D_runner_param.toTensorCI2Param());
        
        tci.addPivotsAllBonds(id_additionnal_pivot);

        for (int it = 0; it < tci_param.nb_iter; ++it) {
            tci.iterate();

            std::ostringstream oss_pivot;

            oss_pivot << std::scientific
                    << std::setprecision(std::numeric_limits<Scalar>::max_digits10);

            oss_pivot << "sweep " << (it + 1)
                    << " |pivot error| = "
                    << tci.last_pivot_error();

            log(oss_pivot.str());
        }
        TensorTrain<Complex> tt_temp = tci.get_TensorTrain();
        std::vector<Cube<Complex>> cube_cores = tt_temp.convert_to_cubes();

        save_vec_cube(cube_cores, file_prefix + ".tt");

        log("Max bond dim: " + std::to_string(tt_temp.max_bond_dimension()));
        
        log("Nb function call:" + std::to_string(counter));
        log("Nb function cached call:" + std::to_string(counter_cached));

        TTErrorOnGrid<Scalar> error = 
            error_TT_on_grid_point(tt_temp, function_id, grid, nb_point_res);
        
        std::ostringstream oss;
        oss << error;
        log(oss.str());

        std::vector<Scalar> pivot_error = tci.pivotErrors();

        // save the tt and the grid
        if (do_save){
            save_TTErrorOnGrid(error, file_prefix + "_error.dat");
            grid.save_json(file_prefix);
        }
        // makeCanonical
    }


};
