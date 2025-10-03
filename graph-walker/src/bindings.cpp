#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "randomWalks.hpp"
#include "anonymize.hpp"
#include "directions.hpp"
#include "edgeTypes.hpp"
#include "asText.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_walker, m)
{
    m.def("random_walks", &randomWalks, "random walks");
    m.def("random_walks_with_restart", &randomWalksRestart, "random walks with restart");
    m.def("random_walks_with_no_backtrack", &randomWalksNoBacktrack, "random walks with no backtracking");
    m.def("random_walks_with_restart_no_backtrack", &randomWalksRestartNoBacktrack, "random walks with restart and no backtracking");
    m.def("random_walks_with_periodic_restart", &randomWalksPeriodicRestart, "random walks with periodic restart");
    m.def("random_walks_with_periodic_restart_no_backtrack", &randomWalksPeriodicRestartNoBacktrack, "random walks with periodic restart and no backtracking");
    m.def("node2vec_random_walks", &n2vRandomWalks, "node2vec random walks");
    m.def("anonymize", &anonymize, "anonymize random walks");
    m.def("anonymize_edge_types", &anonymize_edge_types, "anonymize edge types in random walks");
    m.def("anonymize_with_neighbors", &anonymizeNeighbors, "anonymize random walks with neighbors");
    m.def("parse_directions", &parseDirections, "parse directions of random walks");
    m.def("parse_directions_with_neighbors", &parseDirectionsNeighbors, "parse directions of random walks with neighbors");
    m.def("parse_edge_types_and_directions", &parseEdgeTypesAndDirections, "parse edge types and directions of random walks");
    m.def("parse_edge_types_and_directions_with_neighbors", &parseEdgeTypesAndDirectionsNeighbors, "parse edge types and directions of random walks with neighbors");
    m.def("as_text", &asText, "convert random walks to text");
    m.def("as_text_with_neighbors", &asTextNeighbors, "convert random walks to text with neighbors");
}
