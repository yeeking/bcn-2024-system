//
// tokenizer.m
//
// This file was automatically generated and should not be edited.
//

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "tokenizer.h"

@implementation tokenizerInput

- (instancetype)initWithHidden_state_1:(MLMultiArray *)hidden_state_1 x_1:(MLMultiArray *)x_1 {
    self = [super init];
    if (self) {
        _hidden_state_1 = hidden_state_1;
        _x_1 = x_1;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"hidden_state_1", @"x_1"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"hidden_state_1"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.hidden_state_1];
    }
    if ([featureName isEqualToString:@"x_1"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.x_1];
    }
    return nil;
}

@end

@implementation tokenizerOutput

- (instancetype)initWithLinear_21:(MLMultiArray *)linear_21 {
    self = [super init];
    if (self) {
        _linear_21 = linear_21;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"linear_21"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"linear_21"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.linear_21];
    }
    return nil;
}

@end

@implementation tokenizer


/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"tokenizer" ofType:@"mlmodelc"];
    if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load tokenizer.mlmodelc in the bundle resource"); return nil; }
    return [NSURL fileURLWithPath:assetPath];
}


/**
    Initialize tokenizer instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of tokenizer.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model {
    self = [super init];
    if (!self) { return nil; }
    _model = model;
    if (_model == nil) { return nil; }
    return self;
}


/**
    Initialize tokenizer instance with the model in this bundle.
*/
- (nullable instancetype)init {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}


/**
    Initialize tokenizer instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
}


/**
    Initialize tokenizer instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for tokenizer.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Initialize tokenizer instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for tokenizer.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Construct tokenizer instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid tokenizer instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(tokenizer * _Nullable model, NSError * _Nullable error))handler {
    [self loadContentsOfURL:(NSURL * _Nonnull)[self URLOfModelInThisBundle]
              configuration:configuration
          completionHandler:handler];
}


/**
    Construct tokenizer instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid tokenizer instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(tokenizer * _Nullable model, NSError * _Nullable error))handler {
    [MLModel loadContentsOfURL:modelURL
                 configuration:configuration
             completionHandler:^(MLModel *model, NSError *error) {
        if (model != nil) {
            tokenizer *typedModel = [[tokenizer alloc] initWithMLModel:model];
            handler(typedModel, nil);
        } else {
            handler(nil, error);
        }
    }];
}

- (nullable tokenizerOutput *)predictionFromFeatures:(tokenizerInput *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self predictionFromFeatures:input options:[[MLPredictionOptions alloc] init] error:error];
}

- (nullable tokenizerOutput *)predictionFromFeatures:(tokenizerInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [self.model predictionFromFeatures:input options:options error:error];
    if (!outFeatures) { return nil; }
    return [[tokenizerOutput alloc] initWithLinear_21:(MLMultiArray *)[outFeatures featureValueForName:@"linear_21"].multiArrayValue];
}

- (void)predictionFromFeatures:(tokenizerInput *)input completionHandler:(void (^)(tokenizerOutput * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (!prediction) {
            completionHandler(nil, predictionError);
        } else {
            tokenizerOutput *output = [[tokenizerOutput alloc] initWithLinear_21:(MLMultiArray *)[prediction featureValueForName:@"linear_21"].multiArrayValue];
            completionHandler(output, predictionError);
        }
    }];
}

- (void)predictionFromFeatures:(tokenizerInput *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(tokenizerOutput * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input options:options completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (!prediction) {
            completionHandler(nil, predictionError);
        } else {
            tokenizerOutput *output = [[tokenizerOutput alloc] initWithLinear_21:(MLMultiArray *)[prediction featureValueForName:@"linear_21"].multiArrayValue];
            completionHandler(output, predictionError);
        }
    }];
}

- (nullable tokenizerOutput *)predictionFromHidden_state_1:(MLMultiArray *)hidden_state_1 x_1:(MLMultiArray *)x_1 error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    tokenizerInput *input_ = [[tokenizerInput alloc] initWithHidden_state_1:hidden_state_1 x_1:x_1];
    return [self predictionFromFeatures:input_ error:error];
}

- (nullable NSArray<tokenizerOutput *> *)predictionsFromInputs:(NSArray<tokenizerInput*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLBatchProvider> inBatch = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
    id<MLBatchProvider> outBatch = [self.model predictionsFromBatch:inBatch options:options error:error];
    if (!outBatch) { return nil; }
    NSMutableArray<tokenizerOutput*> *results = [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
    for (NSInteger i = 0; i < outBatch.count; i++) {
        id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
        tokenizerOutput * result = [[tokenizerOutput alloc] initWithLinear_21:(MLMultiArray *)[resultProvider featureValueForName:@"linear_21"].multiArrayValue];
        [results addObject:result];
    }
    return results;
}

@end
