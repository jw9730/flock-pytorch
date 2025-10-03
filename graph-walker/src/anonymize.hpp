#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<uint32_t> anonymize(
    py::array_t<uint32_t> _walks);
py::array_t<uint32_t> anonymize_edge_types(
    py::array_t<uint32_t> _walks);
std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<bool>, py::array_t<bool>> anonymizeNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices);
