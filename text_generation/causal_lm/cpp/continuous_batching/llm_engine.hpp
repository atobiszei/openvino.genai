// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"


template <typename T>
void print_array(T * array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    if (tensor.get_element_type() == ov::element::i32) {
        print_array(tensor.data<int>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_array(tensor.data<int64_t>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_array(tensor.data<float>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_array(tensor.data<bool>(), tensor.get_size());
    }
}

struct GenerationResult {
    // request ID
    uint64_t m_request_id;
    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<TokenIds> m_generation_ids;
    // score (cumulative logprob)
    float m_cumulative_logprob;

    static GenerationResult from_sequence_group(const SequenceGroup& sequence_group) {
        GenerationResult result;
        result.m_request_id = sequence_group.get_request_id();

        for (size_t sequence_id = 0; sequence_id < sequence_group.num_finished_seqs(); ++sequence_id) {
            result.m_generation_ids.push_back(sequence_group[sequence_id].get_generated_ids());
        }

        // TODO: track this information
        result.m_cumulative_logprob = 0.0f;

        return result;
    }
};

class LLMEngine {
    CacheManager m_cache_manager;
    Scheduler m_scheduler;
    ModelRunner m_model_runner;
    Sampler m_sampler;

    // current requests to process
    std::vector<SequenceGroup> m_requests;

    void _free_finished_groups() {
        std::remove_if(m_requests.begin(), m_requests.end(), [] (const SequenceGroup& seq_group) {
            return seq_group.has_finished();
        });
    }
public:
    LLMEngine(ov::InferRequest& request,
              const SchedulerConfig& scheduler_config)
        : m_scheduler(scheduler_config),
          m_model_runner(request) {
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_cache_manager.get_num_layers(); ++decoder_layer_id) {
            request.set_input_tensor(2 + decoder_layer_id * 2, m_cache_manager.get_key_cache(decoder_layer_id));
            request.set_input_tensor(2 + decoder_layer_id * 2 + 1, m_cache_manager.get_value_cache(decoder_layer_id));
        }
    }

    void add_request(uint64_t request_id, const TokenIds input_ids, SamplingParameters sampling_params) {
        SequenceGroup sequence_group(request_id, input_ids, sampling_params);
        m_requests.push_back(sequence_group);
    }

    void add_request(uint64_t request_id, const ov::Tensor input_ids, SamplingParameters sampling_params) {
        SequenceGroup sequence_group(request_id, input_ids, sampling_params);
        m_requests.push_back(sequence_group);
    }

    std::vector<GenerationResult> step() {
        Scheduler::Output scheduler_output = m_scheduler.schedule(m_requests);
        m_cache_manager.copy_blocks(scheduler_output.m_block_copy_map);
        ov::Tensor logits = m_model_runner.step(m_requests, scheduler_output);
        m_sampler.decode(m_requests, logits);

        // perform post-processing of current step

        std::vector<GenerationResult> currently_finished_requests;
        for (size_t request_id = 0; request_id < m_requests.size(); ++request_id) {
            const SequenceGroup& sequence_group = m_requests[request_id];
            if (sequence_group.has_finished()) {
                currently_finished_requests.push_back(GenerationResult::from_sequence_group(sequence_group));
            }
        }

        _free_finished_groups();

        return currently_finished_requests;
    }

    bool has_unfinished_requests() const {
        for (auto & sequence_group : m_requests) {
            if (!sequence_group.has_finished())
                return true;
        }

        return false;
    }

    // more high level interface
    std::vector<GenerationResult> generate(const std::vector<ov::Tensor> prompts, std::vector<SamplingParameters> sampling_params) {
        OPENVINO_ASSERT(prompts.size() == sampling_params.size());

        for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
            add_request(request_id, prompts[request_id], sampling_params[request_id]);
        }

        std::vector<GenerationResult> results;
        results.reserve(m_requests.size());

        while (has_unfinished_requests()) {
            std::vector<GenerationResult> partial_results = step();
            results.insert(results.end(), partial_results.begin(), partial_results.end());
        }

        // sort results according to request_id to return results in order of initial prompts
        std::sort(results.begin(), results.end(), [] (const GenerationResult& r1, const GenerationResult& r2) -> bool {
            return r1.m_request_id < r2.m_request_id;
        });

        return results;
    }
};
