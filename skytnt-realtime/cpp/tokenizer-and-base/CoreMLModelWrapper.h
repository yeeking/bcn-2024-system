#pragma once
#include <string>
#include <vector>

class CoreMLModelWrapper {
public:
    CoreMLModelWrapper();
    ~CoreMLModelWrapper();

    // Load the models from the specified path
    bool loadModel(const std::string& modelPath);

    // Predict function that chains the base model and tokenizer
    bool predict(const std::vector<float>& inputBase, const std::vector<float>& inputX, std::vector<float>& output);

    // Functions to get input and output vectors based on the model's schema
    std::vector<float> getInputVector();
    std::vector<float> getOutputVector();

private:
    // Hide the Objective-C implementation details (forward declare the model pointer)
    struct Impl;
    Impl* impl_;
};
