#include "newmodel.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

int main() {
    @autoreleasepool {
        // Initialize the model
        newmodel *model = [[newmodel alloc] init];
        if (!model) {
            NSLog(@"Error loading model");
            return -1;
        }

        // Generate some dummy input data (1x16x8) for the model
        NSError *error = nil;
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:@[@1, @16, @8]
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
        if (error) {
            NSLog(@"Error creating MLMultiArray: %@", error.localizedDescription);
            return -1;
        }

        // Fill the input array with some random data
        for (int i = 0; i < inputArray.count; ++i) {
            inputArray[i] = @(arc4random_uniform(100) / 100.0f); // Random float between 0 and 1
        }

        // Perform prediction
        newmodelOutput *output = [model predictionFromX_1:inputArray error:&error];
        if (error || !output) {
            NSLog(@"Error during prediction: %@", error.localizedDescription);
            return -1;
        }

        // Access the output (var_1661)
        MLMultiArray *outputArray = output.var_1661;
        NSLog(@"Prediction result: %@", outputArray);
    }

    return 0;
}
