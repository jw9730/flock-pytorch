#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>> parseEdgeTypesAndDirections(
    py::array_t<uint32_t> _walks,
    py::array_t<bool> _restarts,
    py::array_t<uint32_t> _indptr_edge_type,
    py::array_t<uint32_t> _indices_edge_type,
    py::array_t<uint32_t> _data_edge_type,
    py::array_t<uint32_t> _indptr_edge_type_transposed,
    py::array_t<uint32_t> _indices_edge_type_transposed,
    py::array_t<uint32_t> _data_edge_type_transposed,
    size_t seed);

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>> parseEdgeTypesAndDirectionsNeighbors(
    py::array_t<uint32_t> _walks, // walk matrix
    py::array_t<bool> _restarts,  // restart binary mask
    py::array_t<bool> _neighbors, // neighbor binary mask
    /* Edge matrix */
    py::array_t<uint32_t> _indptr_edge_type,
    py::array_t<uint32_t> _indices_edge_type,
    py::array_t<uint32_t> _data_edge_type,
    /* Transposed edge matrix */
    py::array_t<uint32_t> _indptr_edge_type_transposed,
    py::array_t<uint32_t> _indices_edge_type_transposed,
    py::array_t<uint32_t> _data_edge_type_transposed,
    size_t seed
);
