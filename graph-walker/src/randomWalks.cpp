#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "randomWalks.hpp"
#include "threading.hpp"

namespace py = pybind11;

py::array_t<uint32_t> randomWalks(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    size_t seed,
    size_t nWalks,
    size_t walkLen)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we fill in current node
            if (start == end)
            {
                walks[i * walkLen + k] = step;
                continue;
            }

            // searchsorted
            float cumsum = 0;
            size_t index = 0;
            float draw = draws[k - 1];
            for (size_t z = start; z < end; z++)
            {
                cumsum += data[z];
                if (draw > cumsum)
                {
                    continue;
                }
                else
                {
                    index = z;
                    break;
                }
            }

            // draw next index
            step = indices[index];

            // update walk
            walks[i * walkLen + k] = step;
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

py::array_t<uint32_t> randomWalksNoBacktrack(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    size_t seed,
    size_t nWalks,
    size_t walkLen)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we fill in current node
            if (start == end)
            {
                walks[i * walkLen + k] = step;
                continue;
            }

            if (k >= 2)
            {
                uint32_t prev = walks[i * walkLen + k - 2];
                uint32_t prevStart = indptr[prev];
                uint32_t prevEnd = indptr[prev + 1];

                float weightSum = 0.;
                std::vector<float> weights;
                weights.reserve(end - start);

                for (size_t z = start; z < end; z++)
                {
                    uint32_t neighbor = indices[z];
                    float weight = data[z];
                    if (neighbor == prev)
                    {
                        // case where candidate is the previous node
                        weight = 0;
                    }
                    weights[z - start] = weight;
                    weightSum += weight;
                }

                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1] * weightSum;
                for (size_t z = 0; z < end - start; z++)
                {
                    cumsum += weights[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }
                // select next index
                step = indices[index + start];

                // update walk
                walks[i * walkLen + k] = step;

            }
            else
            {
                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1];
                for (size_t z = start; z < end; z++)
                {
                    cumsum += data[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }

                // draw next index
                step = indices[index];

                // update walk
                walks[i * walkLen + k] = step;
            }
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

std::tuple<py::array_t<uint32_t>, py::array_t<bool>> randomWalksRestart(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    py::array_t<bool> _prefixRestart,
    size_t seed,
    size_t nWalks,
    size_t walkLen,
    float alpha)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    py::buffer_info prefixRestartBuf = _prefixRestart.request();
    bool *prefixRestart = (bool *)prefixRestartBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // restart matrix
    py::array_t<bool> _restarts({shape, walkLen});
    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        bool restart;
        size_t startNode = prefix[(i % nPrefixes) * prefixLen];
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            restart = prefixRestart[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
            restarts[i * walkLen + k] = restart;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we restart or fill in current node
            if (start == end)
            {
                if (dist(generator) < alpha)
                {
                    step = startNode;
                    restart = true;
                }
                walks[i * walkLen + k] = step;
                restarts[i * walkLen + k] = restart;
                continue;
            }

            // do not restart if restarted in previous step
            if (!restarts[i * walkLen + k - 1] && (dist(generator) < alpha))
            {
                step = startNode;
                restart = true;
            }
            else
            {
                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1];
                for (size_t z = start; z < end; z++)
                {
                    cumsum += data[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }

                // draw next index
                step = indices[index];
                restart = false;
            }

            // update walk
            walks[i * walkLen + k] = step;
            restarts[i * walkLen + k] = restart;
        }
    }
    PARALLEL_FOR_END();

    return {_walks, _restarts};
}

std::tuple<py::array_t<uint32_t>, py::array_t<bool>> randomWalksRestartNoBacktrack(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    py::array_t<bool> _prefixRestart,
    size_t seed,
    size_t nWalks,
    size_t walkLen,
    float alpha)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    py::buffer_info prefixRestartBuf = _prefixRestart.request();
    bool *prefixRestart = (bool *)prefixRestartBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // restart matrix
    py::array_t<bool> _restarts({shape, walkLen});
    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        bool restart;
        size_t startNode = prefix[(i % nPrefixes) * prefixLen];
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            restart = prefixRestart[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
            restarts[i * walkLen + k] = restart;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we restart or fill in current node
            if (start == end)
            {
                if (dist(generator) < alpha)
                {
                    step = startNode;
                    restart = true;
                }
                walks[i * walkLen + k] = step;
                restarts[i * walkLen + k] = restart;
                continue;
            }

            if (k >= 2)
            {
                // do not restart if restarted in previous step
                if (!restarts[i * walkLen + k - 1] && (dist(generator) < alpha))
                {
                    step = startNode;
                    restart = true;
                }
                else
                {
                    uint32_t prev = walks[i * walkLen + k - 2];
                    uint32_t prevStart = indptr[prev];
                    uint32_t prevEnd = indptr[prev + 1];

                    float weightSum = 0.;
                    std::vector<float> weights;
                    weights.reserve(end - start);

                    for (size_t z = start; z < end; z++)
                    {
                        uint32_t neighbor = indices[z];
                        float weight = data[z];
                        if (neighbor == prev)
                        {
                            // case where candidate is the previous node
                            weight = 0;
                        }
                        weights[z - start] = weight;
                        weightSum += weight;
                    }

                    // searchsorted
                    float cumsum = 0;
                    size_t index = 0;
                    float draw = draws[k - 1] * weightSum;
                    for (size_t z = 0; z < end - start; z++)
                    {
                        cumsum += weights[z];
                        if (draw > cumsum)
                        {
                            continue;
                        }
                        else
                        {
                            index = z;
                            break;
                        }
                    }
                    // select next index
                    step = indices[index + start];
                    restart = false;
                }

                // update walk
                walks[i * walkLen + k] = step;
                restarts[i * walkLen + k] = restart;
            }
            else
            {
                // do not restart if restarted in previous step
                if (!restarts[i * walkLen + k - 1] && (dist(generator) < alpha))
                {
                    step = startNode;
                    restart = true;
                }
                else
                {
                    // searchsorted
                    float cumsum = 0;
                    size_t index = 0;
                    float draw = draws[k - 1];
                    for (size_t z = start; z < end; z++)
                    {
                        cumsum += data[z];
                        if (draw > cumsum)
                        {
                            continue;
                        }
                        else
                        {
                            index = z;
                            break;
                        }
                    }

                    // draw next index
                    step = indices[index];
                    restart = false;
                }

                // update walk
                walks[i * walkLen + k] = step;
                restarts[i * walkLen + k] = restart;
            }
        }
    }
    PARALLEL_FOR_END();

    return {_walks, _restarts};
}

py::array_t<uint32_t> randomWalksPeriodicRestart(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    size_t seed,
    size_t nWalks,
    size_t walkLen,
    size_t period)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        size_t startNode = prefix[(i % nPrefixes) * prefixLen];
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we restart or fill in current node
            if (start == end)
            {
                if (k % period == 0)
                {
                    step = startNode;
                }
                walks[i * walkLen + k] = step;
                continue;
            }

            if (k % period == 0)
            {
                step = startNode;
            }
            else
            {
                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1];
                for (size_t z = start; z < end; z++)
                {
                    cumsum += data[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }

                // draw next index
                step = indices[index];
            }

            // update walk
            walks[i * walkLen + k] = step;
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

py::array_t<uint32_t> randomWalksPeriodicRestartNoBacktrack(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    size_t seed,
    size_t nWalks,
    size_t walkLen,
    size_t period)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        size_t startNode = prefix[(i % nPrefixes) * prefixLen];
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we restart or fill in current node
            if (start == end)
            {
                if (k % period == 0)
                {
                    step = startNode;
                }
                walks[i * walkLen + k] = step;
                continue;
            }

            if (k >= 2)
            {
                if (k % period == 0)
                {
                    step = startNode;
                }
                else
                {
                    uint32_t prev = walks[i * walkLen + k - 2];

                    float weightSum = 0.;
                    std::vector<float> weights;
                    weights.reserve(end - start);

                    for (size_t z = start; z < end; z++)
                    {
                        uint32_t neighbor = indices[z];
                        float weight = data[z];
                        if (neighbor == prev)
                        {
                            // case where candidate is the previous node
                            weight = 0;
                        }
                        weights[z - start] = weight;
                        weightSum += weight;
                    }

                    // searchsorted
                    float cumsum = 0;
                    size_t index = 0;
                    float draw = draws[k - 1] * weightSum;
                    for (size_t z = 0; z < end - start; z++)
                    {
                        cumsum += weights[z];
                        if (draw > cumsum)
                        {
                            continue;
                        }
                        else
                        {
                            index = z;
                            break;
                        }
                    }
                    // select next index
                    step = indices[index + start];
                }

                // update walk
                walks[i * walkLen + k] = step;
            }
            else
            {
                if (k % period == 0)
                {
                    step = startNode;
                }
                else
                {
                    // searchsorted
                    float cumsum = 0;
                    size_t index = 0;
                    float draw = draws[k - 1];
                    for (size_t z = start; z < end; z++)
                    {
                        cumsum += data[z];
                        if (draw > cumsum)
                        {
                            continue;
                        }
                        else
                        {
                            index = z;
                            break;
                        }
                    }

                    // draw next index
                    step = indices[index];
                }

                // update walk
                walks[i * walkLen + k] = step;
            }
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

py::array_t<uint32_t> n2vRandomWalks(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _prefix,
    size_t seed,
    size_t nWalks,
    size_t walkLen,
    float p,
    float q)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info prefixBuf = _prefix.request();
    uint32_t *prefix = (uint32_t *)prefixBuf.ptr;

    // general variables
    size_t nPrefixes = prefixBuf.shape[0];
    size_t prefixLen = prefixBuf.shape[1];
    size_t shape = nWalks * nPrefixes;
    if (prefixLen > walkLen)
    {
        throw std::runtime_error("prefixLen cannot be greater than walkLen");
    }

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step;
        for (size_t k = 0; k < prefixLen; k++)
        {
            step = prefix[(i % nPrefixes) * prefixLen + k];
            walks[i * walkLen + k] = step;
        }

        for (size_t k = prefixLen; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we fill in current node
            if (start == end)
            {
                walks[i * walkLen + k] = step;
                continue;
            }

            if (k >= 2)
            {
                uint32_t prev = walks[i * walkLen + k - 2];
                uint32_t prevStart = indptr[prev];
                uint32_t prevEnd = indptr[prev + 1];

                float weightSum = 0.;
                std::vector<float> weights;
                weights.reserve(end - start);
                for (size_t z = start; z < end; z++)
                {
                    uint32_t neighbor = indices[z];
                    float weight = data[z];
                    if (neighbor == prev)
                    {
                        // case where candidate is the previous node
                        weight /= p;
                    }
                    else
                    {
                        // check if candidate is a neighbor of previous node
                        bool isInPrev = false;
                        for (size_t pi = prevStart; pi < prevEnd; pi++)
                        {
                            if (neighbor != indices[pi])
                                continue;
                            isInPrev = true;
                            break;
                        }
                        if (!isInPrev)
                            weight /= q;
                    }
                    weights[z - start] = weight;
                    weightSum += weight;
                }

                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1] * weightSum;
                for (size_t z = 0; z < end - start; z++)
                {
                    cumsum += weights[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }
                // select next index
                step = indices[index + start];

                // update walk
                walks[i * walkLen + k] = step;
            }
            else
            {
                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1];
                for (size_t z = start; z < end; z++)
                {
                    cumsum += data[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }

                // select next index
                step = indices[index];

                // update walk
                walks[i * walkLen + k] = step;
            }
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}
