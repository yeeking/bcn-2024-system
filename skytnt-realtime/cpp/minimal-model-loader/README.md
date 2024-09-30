Compile the model exported from python:

```
xcrun coremlc compile ../../python/newmodel.mlpackage
xcrun coremlc generate ../../python/newmodel.mlpackage .
```

Build MyApp from source:

```
clang++  -fobjc-arc  -framework Foundation -framework CoreML -o MyApp myk.mm newmodel.m
#Or with cmake
cmake -B build .
cmake --build build 
```


