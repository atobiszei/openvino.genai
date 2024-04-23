// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include "synchronized_queue.hpp"
#include "generation_handle.hpp"


class GenerationStream {
    std::mutex m_mutex;
    bool m_generation_finished = false;
    SynchronizedQueue<GenerationOutputs> m_output_queue;

    std::vector<uint64_t> last_sequence_ids;

public:
    using Ptr = std::shared_ptr<GenerationStream>;

    // Don't use directly
    GenerationStream() = default;

    static GenerationStream::Ptr create() {
        return std::make_shared<GenerationStream>();
    }

    void push(GenerationOutputs outputs) {
        m_output_queue.push(outputs);
    }

    // Retriving vector of pairs <sequence_id, token_id> as we can generate multiple outputs for a single prompt
    GenerationOutputs read() {
        return m_output_queue.pull();
    }

    bool can_read() {
        return !m_output_queue.empty();
    }

    void finish_generation_stream() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_generation_finished = true;
    }

    bool generation_finished() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_generation_finished;
    }
};
