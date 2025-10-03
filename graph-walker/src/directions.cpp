#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <queue>
#include "directions.hpp"
#include "threading.hpp"

namespace py = pybind11;

py::array_t<bool> parseDirections(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // direction matrix
    py::array_t<bool> _backwards({shape, walkLen});
    py::buffer_info backwardsBuf = _backwards.request();
    bool *backwards = (bool *)backwardsBuf.ptr;

    // parse directions
    PARALLEL_FOR_BEGIN(shape)
    {
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];

            // if not the first node and not a restart, check direction
            if (k > 0 && !restart)
            {
                size_t prev = walks[i * walkLen + k - 1];
                size_t start = indptr[prev];
                size_t end = indptr[prev + 1];
                bool found = false;
                for (size_t z = start; z < end; z++)
                {
                    if (indices[z] == value)
                    {
                        found = true;
                        break;
                    }
                }
                backwards[i * walkLen + k] = !found;
            }
            else
            {
                backwards[i * walkLen + k] = false;
            }
        }
    }
    PARALLEL_FOR_END();

    return _backwards;
}


py::array_t<bool> parseDirectionsNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors,
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // direction matrix
    py::array_t<bool> _backwards({shape, walkLen});
    py::buffer_info backwardsBuf = _backwards.request();
    bool *backwards = (bool *)backwardsBuf.ptr;

    // parse directions
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t prev = walks[i * walkLen];
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            bool neighbor = neighbors[i * walkLen + k];

            // if not the first node and not a restart, check direction
            if (k > 0 && !restart)
            {
                size_t start = indptr[prev];
                size_t end = indptr[prev + 1];
                bool found = false;
                for (size_t z = start; z < end; z++)
                {
                    if (indices[z] == value)
                    {
                        found = true;
                        break;
                    }
                }
                backwards[i * walkLen + k] = !found;
            }
            else
            {
                backwards[i * walkLen + k] = false;
            }

            // if not a neighbor, update prev
            if (!neighbor)
            {
                prev = value;
            }
        }
    }
    PARALLEL_FOR_END();

    return _backwards;
}
