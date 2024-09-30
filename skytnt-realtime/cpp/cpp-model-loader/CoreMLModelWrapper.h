#pragma once
#include <string>
#include <vector>

class CoreMLModelWrapper {
public:
    CoreMLModelWrapper();
    ~CoreMLModelWrapper();

    // The model path is now passed as an argument to loadModel
    bool loadModel(const std::string& modelPath);
    /** pass data through the model and return the result */
    std::vector<float> predict(const std::vector<float>& inputData);
    /** pass data and write the result to a sent output buffer */
    bool predict(const std::vector<float>& inputData, std::vector<float>& outputData);
    /** get a vector that is the right shape for input data */
    std::vector<float> getInputVector();
    /** get a vector that is the right shape for output data */
    std::vector<float> getOutputVector();
private:
    // Hide the Objective-C implementation details (forward declare the model pointer)
    struct Impl;
    Impl* impl_;
};
