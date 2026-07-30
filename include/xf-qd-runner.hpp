
#include <grid.h>

template <typename Scalar, typename Sint>
class TCI_Runner{
    public:
    
    using Complex = std::complex<Scalar>;

    const QTGrid<Scalar, Sint> grid;
    const TCI_param tci_param;
    std::function<Complex(Scalar)> function_x;  // original function
    std::function<Complex(MultiIndex)> function_id; // function on grid
    const std::vector<int> l_d;
    int counter = 0;
    int counter_cached = 0;
    std::shared_ptr<spdlog::logger> const logger;

    
    TCI_Runner(
        QTGrid<Scalar, Sint> grid_,
        TCI_param tci_param_,
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

    std::function<Complex(MultiIndex)>
    func_to_grid(
        std::function<Complex(Scalar)> f,
        const QTGrid<Scalar, Sint>& qt_grid, // be carefull of lifetime. The output function can't outlive the TCI_runner class. A full copy would be totaly free.
        std::shared_ptr<spdlog::logger> logger_ = nullptr
    )
    {
        if (logger_){
                return [this, &qt_grid, f](const MultiIndex& id) -> Complex {
                    counter++;
                    return f(qt_grid.id_to_coord(id.data));
                };
            }
        else{
            return [&qt_grid, f](const MultiIndex& id) -> Complex {
                    return f(qt_grid.id_to_coord(id.data));
                };
        }
    }

    std::function<Complex(MultiIndex)>
    make_cached_function(std::function<Complex(MultiIndex)> f)
    {
        auto cache = std::make_shared<std::unordered_map<MultiIndex, Complex, MultiIndexHash>>();

        return [this, f = std::move(f), cache](const MultiIndex& v) mutable -> Complex
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

    std::function<Complex(MultiIndex)>
    resolve_function(
        bool do_cache,
        std::function<Complex(Scalar)> function_x_,
        const QTGrid<Scalar, Sint>& grid_, // be carefull of lifetime. The output function can't outlive the TCI_runner class. A full copy would be totaly free.
        std::shared_ptr<spdlog::logger> logger_ = nullptr
    ){
        std::function<Complex(MultiIndex)> function_on_grid = func_to_grid(function_x_, grid_, logger_);
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
        const Scalar E_init,
        const std::vector<Scalar> other_E,
        bool verbose = true,
        bool do_save=false,
        const std::string& file_prefix="",
        int nb_point_res=1000,
        const std::vector<Scalar> E_discontinuity={},
        const Scalar E_min=0,
        const Scalar E_max=0
    ){
        auto log = [&](const std::string& msg) {
            if (logger) {
                logger->info(msg);
            } 
            if (verbose) {
                std::cout << msg << "\n";
            }
        };

        log("========== FIT PARAMETERS ==========");

        log("E_init = " + to_string_stream(E_init));

        std::string other_E_str = "[";
        for (size_t i = 0; i < other_E.size(); ++i) {
            other_E_str += to_string_stream(other_E[i]);
            if (i + 1 < other_E.size()) other_E_str += ", ";
        }
        other_E_str += "]";

        log("other_E = " + other_E_str);

        log("verbose = " + to_string_stream(verbose ? "true" : "false"));
        log("do_save = " + to_string_stream(do_save ? "true" : "false"));

        log("file_prefix = " + file_prefix);

        log("nb_point_res = " + to_string_stream(nb_point_res));

        log("============== TCI params ==============");
        log("  N = " + std::to_string(grid.get_nBits()));
        log("  d = 2");
        log("  sweeps = " + std::to_string(tci_param.nb_iter));
        log("========================================");

        if (!E_discontinuity.empty()){
            if (E_min !=E_max){
                // if not the case we skip the f_value file generation
                // if E_min and E_max are given, they should be different
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
        }


        log("==========begin iteration===============");

        MultiIndex pivot0 = grid.coord_to_id(E_init);

        TCI<Complex> tci(function_id, l_d, pivot0);
        
        int nb_additional_pivot = other_E.size();

        for (int i=0; i<nb_additional_pivot; i++){
            MultiIndex pivot_temp = grid.coord_to_id(E_init);
            //tci.addpivot_all_bound(pivot_temp);
        }

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
    }


};
