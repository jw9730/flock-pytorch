#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::vector<std::string> asText(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts);
std::vector<std::string> asTextNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors);
