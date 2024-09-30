#include "CoreMLModelWrapper.h"
#include <iostream>
#include <vector>

int main() {
    // Path to the compiled CoreML model (assumed to be in the current directory)
    std::string modelPath = "../newmodel.mlmodelc";

    // Initialize the model wrapper
    CoreMLModelWrapper modelWrapper;

    // Load the model with the specified path
    if (!modelWrapper.loadModel(modelPath)) {
        std::cerr << "Failed to load model from path: " << modelPath << std::endl;
        return -1;
    }

    // Get a suitable input vector of the correct size
    std::vector<float> inputData = modelWrapper.getInputVector();
    // Fill input data with some values (e.g., all 0.5)
    std::fill(inputData.begin(), inputData.end(), 0.5f);

    // Get a suitable output vector
    std::vector<float> outputData = modelWrapper.getOutputVector();

    // Use the overloaded predict method that writes to the pre-allocated output vector
    if (!modelWrapper.predict(inputData, outputData)) {
        std::cerr << "Prediction failed!" << std::endl;
        return -1;
    }

    // Print the result
    // std::cout << "Prediction result:" << std::endl;
    // for (float value : outputData) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
