#include "CoreMLModelWrapper.h"
#include "newmodel.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// Implementation struct to hide Objective-C details
struct CoreMLModelWrapper::Impl {
    newmodel* model;
};

CoreMLModelWrapper::CoreMLModelWrapper() {
    impl_ = new Impl();
}

CoreMLModelWrapper::~CoreMLModelWrapper() {
    delete impl_;
}

bool CoreMLModelWrapper::loadModel(const std::string& modelPath) {
    @autoreleasepool {
        // Convert the C++ string modelPath to an Objective-C NSString and URL
        NSString *modelPathObjC = [NSString stringWithUTF8String:modelPath.c_str()];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPathObjC];
        
        // Load the model from the specified path
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:&error];
        if (error || !model) {
            NSLog(@"Error loading model from path: %@, %@", modelPathObjC, error.localizedDescription);
            return false;
        }
        
        // Initialize the CoreML model object
        impl_->model = [[newmodel alloc] initWithMLModel:model];
        if (!impl_->model) {
            NSLog(@"Error initializing newmodel");
            return false;
        }
    }
    return true;
}

std::vector<float> CoreMLModelWrapper::predict(const std::vector<float>& inputData) {
    @autoreleasepool {
        NSError *error = nil;

        // Convert inputData (std::vector<float>) to MLMultiArray
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:@[@1, @16, @8]
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
        if (error) {
            NSLog(@"Error creating MLMultiArray: %@", error.localizedDescription);
            return {};
        }

        // Fill inputArray with inputData
        for (int i = 0; i < inputArray.count; ++i) {
            inputArray[i] = @(inputData[i]);
        }

        // Perform prediction
        newmodelOutput *output = [impl_->model predictionFromX_1:inputArray error:&error];
        if (error || !output) {
            NSLog(@"Error during prediction: %@", error.localizedDescription);
            return {};
        }

        // Convert MLMultiArray output to std::vector<float>
        MLMultiArray *outputArray = output.var_1661;
        std::vector<float> result(outputArray.count);
        for (int i = 0; i < outputArray.count; ++i) {
            result[i] = [outputArray[i] floatValue];
        }

        return result;
    }
}

bool CoreMLModelWrapper::predict(const std::vector<float>& inputData, std::vector<float>& outputData) {
    @autoreleasepool {
        NSError *error = nil;

        // Convert inputData (std::vector<float>) to MLMultiArray
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:@[@1, @16, @8]
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
        if (error) {
            NSLog(@"Error creating MLMultiArray: %@", error.localizedDescription);
            return false;
        }

        // Fill inputArray with inputData
        for (int i = 0; i < inputArray.count; ++i) {
            inputArray[i] = @(inputData[i]);
        }

        // Perform prediction
        newmodelOutput *output = [impl_->model predictionFromX_1:inputArray error:&error];
        if (error || !output) {
            NSLog(@"Error during prediction: %@", error.localizedDescription);
            return false;
        }

        // Access the MLMultiArray output and convert it to a std::vector
        MLMultiArray *outputArray = output.var_1661;

        // Ensure the outputData vector is properly allocated
        if (outputData.size() != outputArray.count) {
            outputData.resize(outputArray.count);
        }

        // Copy data from MLMultiArray to the output vector
        for (int i = 0; i < outputArray.count; ++i) {
            outputData[i] = [outputArray[i] floatValue];
        }

        return true;
    }
}

std::vector<float> CoreMLModelWrapper::getInputVector() {
    @autoreleasepool {
        // Get model input shape metadata
        NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptions = impl_->model.model.modelDescription.inputDescriptionsByName;
        MLFeatureDescription *inputDesc = inputDescriptions[@"x_1"];
        MLMultiArrayConstraint *inputConstraint = inputDesc.multiArrayConstraint;

        // Calculate the total size based on shape
        NSUInteger totalSize = 1;
        for (NSNumber *dimension in inputConstraint.shape) {
            totalSize *= dimension.unsignedIntegerValue;
        }

        // Return a zero-initialized vector of the correct size
        return std::vector<float>(totalSize, 0.0f);
    }
}

std::vector<float> CoreMLModelWrapper::getOutputVector() {
    @autoreleasepool {
        // Get model output shape metadata
        NSDictionary<NSString *, MLFeatureDescription *> *outputDescriptions = impl_->model.model.modelDescription.outputDescriptionsByName;
        MLFeatureDescription *outputDesc = outputDescriptions[@"var_1661"];
        MLMultiArrayConstraint *outputConstraint = outputDesc.multiArrayConstraint;

        // Calculate the total size based on shape
        NSUInteger totalSize = 1;
        for (NSNumber *dimension in outputConstraint.shape) {
            totalSize *= dimension.unsignedIntegerValue;
        }

        // Return a zero-initialized vector of the correct size
        return std::vector<float>(totalSize, 0.0f);
    }
}