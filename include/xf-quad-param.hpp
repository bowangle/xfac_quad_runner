#pragma once

class TCI_param{
    public:

    int nBit;
    int nb_iter;
    bool do_cache;

    TCI_param(int nBit_, int nb_iter_, bool do_cache_=false)
    :   nBit(nBit_),
        nb_iter(nb_iter_),
        do_cache(do_cache_)
    {}
};
