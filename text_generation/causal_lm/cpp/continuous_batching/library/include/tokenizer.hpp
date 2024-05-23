
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <vector>

#include "openvino/runtime/tensor.hpp"

struct TokenizerConfig {
	std::string chat_template;
	std::string bos_token;
	std::string eos_token;
};

class Tokenizer {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    explicit Tokenizer(const std::string& models_path);

    // note, that returned tensor is shared with internal state of InferRequest
    // so, it can be changed. Please, copy values
    ov::Tensor encode(std::string prompt);

    std::string decode(std::vector<int64_t> tokens);
    const TokenizerConfig& get_config() const;

    size_t get_eos_token_id() const;
};
