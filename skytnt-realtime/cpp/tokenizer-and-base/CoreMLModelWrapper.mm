#include "CoreMLModelWrapper.h"
#include "tokenizer.h"
#include "base.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// Implementation struct to hide Objective-C details
struct CoreMLModelWrapper::Impl {
    tokenizer* model_tokenizer;
    base* model_base;
};

CoreMLModelWrapper::CoreMLModelWrapper() {
    impl_ = new Impl();
}

CoreMLModelWrapper::~CoreMLModelWrapper() {
    delete impl_;
}

bool CoreMLModelWrapper::loadModel(const std::string& modelPath) {
    @autoreleasepool {
        std::string baseModelPath = modelPath + "/base.mlmodelc";
        std::string tokenModelPath = modelPath + "/tokenizer.mlmodelc";
        
        // Convert the C++ string modelPath to an Objective-C NSString and URL for both models
        NSString *baseModelPathObjC = [NSString stringWithUTF8String:baseModelPath.c_str()];
        NSURL *baseModelURL = [NSURL fileURLWithPath:baseModelPathObjC];
        
        // Load the base model
        NSError *error = nil;
        MLModel *baseModel = [MLModel modelWithContentsOfURL:baseModelURL error:&error];
        if (error || !baseModel) {
            NSLog(@"Error loading base model from path: %@, %@", baseModelPathObjC, error.localizedDescription);
            return false;
        }

        NSString *tokenModelPathObjC = [NSString stringWithUTF8String:tokenModelPath.c_str()];
        NSURL *tokenModelURL = [NSURL fileURLWithPath:tokenModelPathObjC];

        // Load the tokenizer model
        MLModel *tokenModel = [MLModel modelWithContentsOfURL:tokenModelURL error:&error];
        if (error || !tokenModel) {
            NSLog(@"Error loading tokenizer model from path: %@, %@", tokenModelPathObjC, error.localizedDescription);
            return false;
        }

        // Initialize the CoreML model objects
        impl_->model_base = [[base alloc] initWithMLModel:baseModel];
        if (!impl_->model_base) {
            NSLog(@"Error initializing base model");
            return false;
        }
        
        impl_->model_tokenizer = [[tokenizer alloc] initWithMLModel:tokenModel];
        if (!impl_->model_tokenizer) {
            NSLog(@"Error initializing tokenizer model");
            return false;
        }
    }
    return true;
}

bool CoreMLModelWrapper::predict(const std::vector<float>& inputBase, const std::vector<float>& inputX, std::vector<float>& output) {
    @autoreleasepool {
        NSError *error = nil;

        // Convert inputBase to MLMultiArray for model_base
        MLMultiArray *inputBaseArray = [[MLMultiArray alloc] initWithShape:@[@(inputBase.size())]
                                                                  dataType:MLMultiArrayDataTypeFloat32
                                                                     error:&error];
        if (error) {
            NSLog(@"Error creating input MLMultiArray for base model: %@", error.localizedDescription);
            return false;
        }

        // Fill inputBaseArray with inputBase data
        for (int i = 0; i < inputBase.size(); ++i) {
            inputBaseArray[i] = @(inputBase[i]);
        }

        // Run the base model and get the hidden state
        baseOutput *baseResult = [impl_->model_base predictionFromX_1:inputBaseArray error:&error];
        if (error || !baseResult) {
            NSLog(@"Error running base model: %@", error.localizedDescription);
            return false;
        }

        // Extract the hidden state (var_hidden) from the base model's output
        MLMultiArray *hiddenState = baseResult.var_1661;  // Assuming `var_hidden` is the hidden state

        // Convert inputX to MLMultiArray for model_tokenizer
        MLMultiArray *inputXArray = [[MLMultiArray alloc] initWithShape:@[@(inputX.size())]
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                  error:&error];
        if (error) {
            NSLog(@"Error creating input MLMultiArray for tokenizer: %@", error.localizedDescription);
            return false;
        }

        // Fill inputXArray with inputX data
        for (int i = 0; i < inputX.size(); ++i) {
            inputXArray[i] = @(inputX[i]);
        }

        // Run the tokenizer model with hidden_state_1 and x_1
        tokenizerOutput *tokenizerResult = [impl_->model_tokenizer predictionFromHidden_state_1:hiddenState x_1:inputXArray error:&error];
        if (error || !tokenizerResult) {
            NSLog(@"Error running tokenizer model: %@", error.localizedDescription);
            return false;
        }

        // Extract the final output from the tokenizerResult
        MLMultiArray *outputArray = tokenizerResult.linear_21;  // Assuming `linear_21` is the result

        // Resize the output vector if necessary
        if (output.size() != outputArray.count) {
            output.resize(outputArray.count);
        }

        // Copy data from MLMultiArray to the output vector
        for (int i = 0; i < outputArray.count; ++i) {
            output[i] = [outputArray[i] floatValue];
        }

        return true;
    }
}


// Function to get a suitably sized input vector based on the base model's input schema
std::vector<float> CoreMLModelWrapper::getInputVector() {
    @autoreleasepool {
        NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptions = impl_->model_base.model.modelDescription.inputDescriptionsByName;
        MLFeatureDescription *inputDesc = inputDescriptions[@"x_1"];  // Adjust to match your model's input name
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

// Function to get a suitably sized output vector based on the tokenizer model's output schema
std::vector<float> CoreMLModelWrapper::getOutputVector() {
    @autoreleasepool {
        NSDictionary<NSString *, MLFeatureDescription *> *outputDescriptions = impl_->model_tokenizer.model.modelDescription.outputDescriptionsByName;
        MLFeatureDescription *outputDesc = outputDescriptions[@"final_output"];  // Adjust to match your model's output name
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
