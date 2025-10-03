#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <queue>
#include "anonymize.hpp"
#include "threading.hpp"

namespace py = pybind11;

py::array_t<uint32_t> anonymize(
    py::array_t<uint32_t> _walks)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // anonymized walk matrix
    py::array_t<uint32_t> _anon_walks({shape, walkLen});
    py::buffer_info anonWalksBuf = _anon_walks.request();
    uint32_t *anon_walks = (uint32_t *)anonWalksBuf.ptr;

    // anonymize random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        std::unordered_map<int, int> valueToId;
        size_t currentId = 1;

        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];

            // if value is not yet in the map, assign a new id
            if (valueToId.find(value) == valueToId.end())
            {
                valueToId[value] = currentId++;
            }

            // update anonymous walk
            anon_walks[i * walkLen + k] = valueToId[value];
        }
    }
    PARALLEL_FOR_END();

    return _anon_walks;
}

// Same as `anonymize`, but ignores restarts markers (-1)
// Do not anonymize restart markers, just return -1 for them
py::array_t<uint32_t> anonymize_edge_types(
    py::array_t<uint32_t> _walks)
{
    // get data buffer
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // general variables
    size_t shape = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];

    // anonymized walk matrix
    py::array_t<uint32_t> _anon_walks({shape, walkLen});
    py::buffer_info anonWalksBuf = _anon_walks.request();
    uint32_t *anon_walks = (uint32_t *)anonWalksBuf.ptr;

    // anonymize random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        std::unordered_map<int, int> valueToId;
        // Add a special case mapping for restart markers (-1)
        valueToId[-1] = -1; // Keep restart markers as -1
        size_t currentId = 1;

        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];

            // if value is not yet in the map, assign a new id
            if (valueToId.find(value) == valueToId.end())
            {
                valueToId[value] = currentId++;
            }

            // update anonymous walk
            anon_walks[i * walkLen + k] = valueToId[value];
        }
    }
    PARALLEL_FOR_END();

    return _anon_walks;
}

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<bool>, py::array_t<bool>> anonymizeNeighbors(
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
    size_t nEdges = indicesBuf.shape[0];

    // anonymized walk matrix
    py::array_t<uint32_t> _anon_walks({shape, walkLen});
    py::buffer_info anonWalksBuf = _anon_walks.request();
    uint32_t *anon_walks = (uint32_t *)anonWalksBuf.ptr;

    // new walk matrix
    py::array_t<uint32_t> _new_walks({shape, walkLen});
    py::buffer_info newWalksBuf = _new_walks.request();
    uint32_t *new_walks = (uint32_t *)newWalksBuf.ptr;

    // new restart matrix
    py::array_t<bool> _new_restarts({shape, walkLen});
    py::buffer_info newRestartsBuf = _new_restarts.request();
    bool *new_restarts = (bool *)newRestartsBuf.ptr;

    // neighbor matrix
    py::array_t<bool> _neighbors({shape, walkLen});
    py::buffer_info neighborsBuf = _neighbors.request();
    bool *neighbors = (bool *)neighborsBuf.ptr;

    // anonymize random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        std::unordered_map<int, int> valueToId;
        std::unordered_map<int, int> idToValue;
        std::vector<bool> visited(nEdges, false);
        size_t currentId = 1;
        size_t index = 0;

        for (size_t k = 0; k < walkLen; k++)
        {
            size_t value = walks[i * walkLen + k];
            bool restart = restarts[i * walkLen + k];

            if (index == walkLen)
            {
                continue;
            }

            // if value is not yet in the map, assign a new id
            if (valueToId.find(value) == valueToId.end())
            {
                valueToId[value] = currentId;
                idToValue[currentId] = value;
                currentId++;

                // update anonymous walk
                anon_walks[i * walkLen + index] = valueToId[value];
                new_walks[i * walkLen + index] = value;
                new_restarts[i * walkLen + index] = restart;
                neighbors[i * walkLen + index] = false;
                index++;

                // visit identified neighbors with unvisited edges
                uint32_t start = indptr[value];
                uint32_t end = indptr[value + 1];
                std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> minQueue;
                size_t prev = (k > 0) ? walks[i * walkLen + k - 1] : 0;
                for (size_t z = start; z < end; z++)
                {
                    size_t neighbor = indices[z];
                    bool unvisited = !visited[z];
                    if (k > 0)
                    {
                        unvisited = unvisited && (prev != neighbor);
                    }
                    if (unvisited && (valueToId.find(neighbor) != valueToId.end()))
                    {
                        visited[z] = true;
                        minQueue.push(valueToId[neighbor]);
                    }
                }
                while (!minQueue.empty())
                {
                    if (index == walkLen)
                    {
                        break;
                    }
                    size_t neighborId = minQueue.top();
                    anon_walks[i * walkLen + index] = neighborId;
                    new_walks[i * walkLen + index] = idToValue[neighborId];
                    new_restarts[i * walkLen + index] = false;
                    neighbors[i * walkLen + index] = true;
                    index++;
                    minQueue.pop();
                }
            }
            else
            {
                // update anonymous walk
                anon_walks[i * walkLen + index] = valueToId[value];
                new_walks[i * walkLen + index] = value;
                new_restarts[i * walkLen + index] = restart;
                neighbors[i * walkLen + index] = false;
                index++;
            }
        }
    }
    PARALLEL_FOR_END();

    return {_anon_walks, _new_walks, _new_restarts, _neighbors};
}
