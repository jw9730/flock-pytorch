#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<bool> parseDirections(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices);
py::array_t<bool> parseDirectionsNeighbors(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<bool> _neighbors,
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices);
