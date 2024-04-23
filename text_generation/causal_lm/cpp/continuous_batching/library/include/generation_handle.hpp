// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include "generation_config.hpp"

struct GenerationOutput {
    std::vector<int64_t> generated_token_ids;
    float score;
};

using GenerationOutputs = std::unordered_map<uint64_t, GenerationOutput>;

class GenerationStream;

class GenerationHandle {
    std::shared_ptr<GenerationStream> m_generation_stream;
    GenerationConfig m_sampling_params;
 
public:
    GenerationHandle(std::shared_ptr<GenerationStream> generation_stream, const GenerationConfig& sampling_params) :
    m_generation_stream(generation_stream),
    m_sampling_params(sampling_params) {};

    bool generation_finished();

    bool can_read();

    // Reads result of a generation for single iteration
    GenerationOutputs read();
    // Reads all generated tokens for all sequences
    std::vector<GenerationOutput> read_all();
};
