#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include "asText.hpp"
#include "threading.hpp"

namespace py = pybind11;

std::vector<std::string> asText(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        for (size_t k = 0; k < walkLen; k++)
        {
            walkStream << walks[i * walkLen + k];
            if (k < walkLen - 1)
            {
                bool restart = restarts[i * walkLen + k + 1];
                walkStream << (restart ? ";" : "-");
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}


std::vector<std::string> asTextNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // text list
    std::vector<std::string> walksText(shape);

    // convert random walks to text
    PARALLEL_FOR_BEGIN(shape)
    {
        std::ostringstream walkStream;
        for (size_t k = 0; k < walkLen; k++)
        {
            walkStream << walks[i * walkLen + k];
            if (k < walkLen - 1)
            {
                bool restart = restarts[i * walkLen + k + 1];
                bool neighbor = neighbors[i * walkLen + k + 1];
                walkStream << (neighbor ? "#" : (restart ? ";" : "-"));
            }
        }
        walksText[i] = walkStream.str();
    }
    PARALLEL_FOR_END();

    return walksText;
}
