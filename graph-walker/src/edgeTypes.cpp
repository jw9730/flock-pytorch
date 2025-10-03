#include <iostream>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <queue>
#include "edgeTypes.hpp"
#include "threading.hpp"

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
    size_t seed)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info indptrEdgeTypeBuf = _indptr_edge_type.request();
    uint32_t *indptr_edge_type = (uint32_t *)indptrEdgeTypeBuf.ptr;

    py::buffer_info indicesEdgeTypeBuf = _indices_edge_type.request();
    uint32_t *indices_edge_type = (uint32_t *)indicesEdgeTypeBuf.ptr;

    py::buffer_info dataEdgeTypeBuf = _data_edge_type.request();
    uint32_t *data_edge_type = (uint32_t *)dataEdgeTypeBuf.ptr;

    py::buffer_info indptrEdgeTypeTransposedBuf = _indptr_edge_type_transposed.request();
    uint32_t *indptr_edge_type_transposed = (uint32_t *)indptrEdgeTypeTransposedBuf.ptr;

    py::buffer_info indicesEdgeTypeTransposedBuf = _indices_edge_type_transposed.request();
    uint32_t *indices_edge_type_transposed = (uint32_t *)indicesEdgeTypeTransposedBuf.ptr;

    py::buffer_info dataEdgeTypeTransposedBuf = _data_edge_type_transposed.request();
    uint32_t *data_edge_type_transposed = (uint32_t *)dataEdgeTypeTransposedBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // edge type matrix
    py::array_t<uint32_t> _edge_type({shape, walkLen});
    py::buffer_info edgeTypeBuf = _edge_type.request();
    uint32_t *edge_type = (uint32_t *)edgeTypeBuf.ptr;

    // direction matrix
    py::array_t<uint32_t> _backwards({shape, walkLen});
    py::buffer_info backwardsBuf = _backwards.request();
    uint32_t *backwards = (uint32_t *)backwardsBuf.ptr;

    // parse edge types
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);
        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];

            // if not the first node and not a restart, check edge type
            // the graph contains multi-edges with different types
            // in this case, randomly select one of the edge types
            std::queue<size_t> edgeTypesQueue;
            std::queue<bool> backwardsQueue;
            if (k > 0 && !restart)
            {
                size_t prev = walks[i * walkLen + k - 1];
                size_t start = indptr_edge_type[prev];
                size_t end = indptr_edge_type[prev + 1];
                for (size_t z = start; z < end; z++)
                {
                    if (indices_edge_type[z] == value)
                    {
                        edgeTypesQueue.push(data_edge_type[z]);
                        if (value == prev)
                        {
                            // if the previous node is the same as the current node, it is a loop
                            backwardsQueue.push(2);
                        }
                        else
                        {
                            // otherwise, it is a downstream edge
                            backwardsQueue.push(0);
                        }
                    }
                }
                start = indptr_edge_type_transposed[prev];
                end = indptr_edge_type_transposed[prev + 1];
                for (size_t z = start; z < end; z++)
                {
                    if (indices_edge_type_transposed[z] == value)
                    {
                        edgeTypesQueue.push(data_edge_type_transposed[z]);
                        if (value == prev)
                        {
                            // if the previous node is the same as the current node, it is a loop
                            backwardsQueue.push(2);
                        }
                        else
                        {
                            // otherwise, it is an upstream edge
                            backwardsQueue.push(1);
                        }
                    }
                }
                size_t numEdgeTypes = edgeTypesQueue.size();
                if (numEdgeTypes > 1)
                {
                    // if multiple edges are found, randomly select one
                    std::uniform_int_distribution<int> dist(0, numEdgeTypes - 1);
                    size_t idx = dist(generator);
                    for (size_t j = 0; j <= idx; j++)
                    {
                        edge_type[i * walkLen + k] = edgeTypesQueue.front();
                        backwards[i * walkLen + k] = backwardsQueue.front();
                        edgeTypesQueue.pop();
                        backwardsQueue.pop();
                    }
                }
                else if (numEdgeTypes == 1)
                {
                    // if a single edge is found, use it
                    edge_type[i * walkLen + k] = edgeTypesQueue.front();
                    backwards[i * walkLen + k] = backwardsQueue.front();
                    edgeTypesQueue.pop();
                    backwardsQueue.pop();
                }
                else
                {
                    // if no edge is found, this should only happen for isolated nodes
                    // // print warning and continue
                    // std::cout << "Warning: No edge found between " << prev << " and " << value << ". This is only expected to happen for isolated nodes." << std::endl;
                    edge_type[i * walkLen + k] = -1;
                    backwards[i * walkLen + k] = -1;
                }
            }
            else
            {
                // no edge type for the first node or restart
                edge_type[i * walkLen + k] = -1;
                backwards[i * walkLen + k] = -1;
            }
        }
    }
    PARALLEL_FOR_END();

    return {_edge_type, _backwards};
}

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
    size_t seed)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info restartsBuf = _restarts.request();
    bool *restarts = (bool *)restartsBuf.ptr;

    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    py::buffer_info indptrEdgeTypeBuf = _indptr_edge_type.request();
    uint32_t *indptr_edge_type = (uint32_t *)indptrEdgeTypeBuf.ptr;

    py::buffer_info indicesEdgeTypeBuf = _indices_edge_type.request();
    uint32_t *indices_edge_type = (uint32_t *)indicesEdgeTypeBuf.ptr;

    py::buffer_info dataEdgeTypeBuf = _data_edge_type.request();
    uint32_t *data_edge_type = (uint32_t *)dataEdgeTypeBuf.ptr;

    py::buffer_info indptrEdgeTypeTransposedBuf = _indptr_edge_type_transposed.request();
    uint32_t *indptr_edge_type_transposed = (uint32_t *)indptrEdgeTypeTransposedBuf.ptr;

    py::buffer_info indicesEdgeTypeTransposedBuf = _indices_edge_type_transposed.request();
    uint32_t *indices_edge_type_transposed = (uint32_t *)indicesEdgeTypeTransposedBuf.ptr;

    py::buffer_info dataEdgeTypeTransposedBuf = _data_edge_type_transposed.request();
    uint32_t *data_edge_type_transposed = (uint32_t *)dataEdgeTypeTransposedBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // edge type matrix
    py::array_t<uint32_t> _edge_type({shape, walkLen});
    py::buffer_info edgeTypeBuf = _edge_type.request();
    uint32_t *edge_type = (uint32_t *)edgeTypeBuf.ptr;

    // direction matrix
    py::array_t<uint32_t> _backwards({shape, walkLen});
    py::buffer_info backwardsBuf = _backwards.request();
    uint32_t *backwards = (uint32_t *)backwardsBuf.ptr;

    // parse edge types
    PARALLEL_FOR_BEGIN(shape)
    {
        size_t thread_seed = seed + i;
        std::mt19937 generator(thread_seed);

        /* "last node" that is not a named neighbor node. */
        size_t predecessor = -1;

        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];
            bool neighbor = neighbors[i * walkLen + k];

            /* Queues that contains edge information related to the predecessor. */
            std::queue<size_t> edgeTypesQueue;
            std::queue<bool> backwardsQueue;

            if (restart || k == 0)
            {
                /* No edge type for the first node or restart. */
                edge_type[i * walkLen + k] = -1;
                backwards[i * walkLen + k] = -1;

                /* Update previous nodes. */
                predecessor = value;
            }
            else
            {
                /* Update predecessor related queues. */
                size_t start = indptr_edge_type[predecessor];
                size_t end = indptr_edge_type[predecessor + 1];
                for (size_t z = start; z < end; z++)
                {
                    if (indices_edge_type[z] == value)
                    {
                        edgeTypesQueue.push(data_edge_type[z]);
                        if (value == predecessor)
                        {
                            // if the previous node is the same as the current node, it is a loop
                            backwardsQueue.push(2);
                        }
                        else
                        {
                            // otherwise, it is a downstream edge
                            backwardsQueue.push(0);
                        }
                    }
                }
                start = indptr_edge_type_transposed[predecessor];
                end = indptr_edge_type_transposed[predecessor + 1];
                for (size_t z = start; z < end; z++)
                {
                    if (indices_edge_type_transposed[z] == value)
                    {
                        edgeTypesQueue.push(data_edge_type_transposed[z]);
                        if (value == predecessor)
                        {
                            // if the previous node is the same as the current node, it is a loop
                            backwardsQueue.push(2);
                        }
                        else
                        {
                            // otherwise, it is an upstream edge
                            backwardsQueue.push(1);
                        }
                    }
                }

                /* Assign edge types and directions. */
                size_t numEdgeTypes = edgeTypesQueue.size();
                if (numEdgeTypes > 1)
                {
                    // if multiple edges are found, randomly select one
                    std::uniform_int_distribution<int> dist(0, numEdgeTypes - 1);
                    size_t idx = dist(generator);
                    for (size_t j = 0; j <= idx; j++)
                    {
                        edge_type[i * walkLen + k] = edgeTypesQueue.front();
                        backwards[i * walkLen + k] = backwardsQueue.front();
                        edgeTypesQueue.pop();
                        backwardsQueue.pop();
                    }
                }
                else if (numEdgeTypes == 1)
                {
                    // if a single edge is found, use it
                    edge_type[i * walkLen + k] = edgeTypesQueue.front();
                    backwards[i * walkLen + k] = backwardsQueue.front();
                    edgeTypesQueue.pop();
                    backwardsQueue.pop();
                }
                else
                {
                    // if no edge is found, this should only happen for isolated nodes
                    // print warning and continue
                    // std::cout << "Warning: No edge found between " << predecessor << " and " << value << ". This is only expected to happen for isolated nodes." << std::endl;
                    edge_type[i * walkLen + k] = -1;
                    backwards[i * walkLen + k] = -1;
                }

                /* Update predecessor nodes if the current node is not a named neighbor. */
                if (!neighbor) {
                    predecessor = value;
                }
            }
        }
    }
    PARALLEL_FOR_END();

    return {_edge_type, _backwards};
}
