// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "generation_handle.hpp"
#include "generation_stream.hpp"

bool GenerationHandle::generation_finished() {
    return m_generation_stream->generation_finished();
}

bool GenerationHandle::can_read() {
    return m_generation_stream->can_read();
}

std::unordered_map<uint64_t, GenerationOutput> GenerationHandle::read() {
    return m_generation_stream->read();
}


void add_partial_result(std::unordered_map<uint64_t, GenerationOutput>& partial_results, std::unordered_map<uint64_t, GenerationOutput>& iteration_results) {
    for (auto& iteration_result: iteration_results) {
        auto partial_result_iter = partial_results.find(iteration_result.first);
        if (partial_result_iter == partial_results.end()) {
            partial_results.emplace(iteration_result.first, iteration_result.second);
        } else {
            partial_result_iter->second.generated_token_ids.push_back(iteration_result.second.generated_token_ids[0]);
            partial_result_iter->second.score = iteration_result.second.score;
        }
    }
}

std::vector<GenerationOutput> GenerationHandle::read_all() {
    std::vector<GenerationOutput> results;
    std::unordered_map<uint64_t, GenerationOutput> partial_results;
    while (!generation_finished() || can_read()) {
        // For unary case there's only one iteration and we get all results in a single read() call
        std::unordered_map<uint64_t, GenerationOutput> iteration_results = read();
        add_partial_result(partial_results, iteration_results);
    }

    for (auto& partial_result: partial_results) {
        results.push_back(partial_result.second);
    }
    return results;
}
