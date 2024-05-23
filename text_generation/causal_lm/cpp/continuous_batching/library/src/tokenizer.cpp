
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <fstream>

#include "nlohmann/json.hpp"
#include "openvino/runtime/core.hpp"

#include "tokenizer.hpp"

class Tokenizer::Impl {
    const size_t TOKENIZER_BATCH_SIZE = 1;
    ov::CompiledModel m_tokenizer, m_detokenizer;
    std::size_t m_eos_token_id;
    TokenizerConfig config;

public:
    explicit Impl(const std::string& models_path) {
        ov::Core core;
        core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

        std::shared_ptr<ov::Model> tokenizer_model = core.read_model(models_path + "/openvino_tokenizer.xml");
        const ov::AnyMap& rt_info = tokenizer_model->get_rt_info();
        OPENVINO_ASSERT(rt_info.find("eos_token_id") != rt_info.end(), "Failed to detect \"eos_token_id\" in openvino_tokenizer.xml runtime information");
        m_eos_token_id = rt_info.at("eos_token_id").as<int64_t>();

        // tokenizer and detokenizer work on CPU only
        m_tokenizer = core.compile_model(
            tokenizer_model, "CPU");
        m_detokenizer = core.compile_model(
            models_path + "/openvino_detokenizer.xml", "CPU");
	// TODO path handling
	// TODO error handling
        std::string tokenizerConfigPath = models_path + "/tokenizer_config.json";
        std::cout << "Tokenizer config json path:" << tokenizerConfigPath << std::endl;
        std::cout << "Tokenizer model xml path:" << (models_path + "/openvino_tokenizer.xml") << std::endl;
        std::ifstream f(models_path + "/tokenizer_config.json");
        nlohmann::json json_data = nlohmann::json::parse(f);

	this->config.bos_token = json_data.value("bos_token", "");
	this->config.eos_token = json_data.value("eos_token", "");
	this->config.chat_template = json_data.value("chat_template", "");
    }

    const TokenizerConfig& get_config() const {
	return this->config;
    }

    ov::Tensor encode(std::string prompt) {
        auto tokenizer_infer_request = m_tokenizer.create_infer_request();
        tokenizer_infer_request.set_input_tensor(ov::Tensor{ov::element::string, {TOKENIZER_BATCH_SIZE}, &prompt});
        tokenizer_infer_request.infer();
        return tokenizer_infer_request.get_tensor("input_ids");
    }

    std::string decode(std::vector<int64_t> tokens) {
        auto detokenizer_infer_request = m_detokenizer.create_infer_request();
        detokenizer_infer_request.set_input_tensor(ov::Tensor{ov::element::i64, {TOKENIZER_BATCH_SIZE, tokens.size()}, tokens.data()});
        detokenizer_infer_request.infer();
        return detokenizer_infer_request.get_output_tensor().data<std::string>()[0];
    }

    size_t get_eos_token_id() const {
        return m_eos_token_id;
    }
};

Tokenizer::Tokenizer(const std::string& models_path) {
    m_impl = std::make_shared<Impl>(models_path);
}

const TokenizerConfig& Tokenizer::get_config() const {
    return this->m_impl->get_config();
}

ov::Tensor Tokenizer::encode(std::string prompt) {
    return m_impl->encode(prompt);
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_impl->decode(tokens);
}

size_t Tokenizer::get_eos_token_id() const {
    return m_impl->get_eos_token_id();
}
