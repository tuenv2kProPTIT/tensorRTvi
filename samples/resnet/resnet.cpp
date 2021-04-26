#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger;
// set dynamic shape inputs
int INPUT_H = 224;
int INPUT_W = 224;
int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
using namespace nvinfer1;
// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
        std::cout<<name<<" "<<size<<std::endl;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    // network:network pointer 
    // weightMap: save weights: mapping-layername and Weights of tensorrt
    // Itensor& input: tensorfeed to layer  batchnorm
    
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    
    float *beta = (float*)weightMap[lname + ".bias"].values;
    
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    
    float *var = (float*)weightMap[lname + ".running_var"].values;
    
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;
    if(len==0){
        std::cout<<lname<<std::endl;
    }
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition* network, ITensor& input,
                              int input_dim,int output_dim,
                              std::map<std::string, Weights>& weightMap, std::string layerName,
                              float eps=1e-5, int dowsample=0){
    
    // input_dim = 64, width = input_dim * output_dim / 64 = output_dim/4
    int stride=1;
    if(dowsample ==0 || dowsample==1){
        stride = stride + dowsample;
    };
    int width = output_dim / 4;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(
        input, width, DimsHW{1,1}, weightMap[layerName + ".conv1.weight"], emptywts );
    assert(conv1);
    conv1->setStrideNd(DimsHW{1,1});
    
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), layerName +".bn1",eps);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    
    
    IConvolutionLayer* conv2 = network->addConvolutionNd(
        *relu1->getOutput(0), width, DimsHW{3,3}, weightMap[layerName + ".conv2.weight"], emptywts
    );
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride,stride});
    conv2->setPaddingNd(DimsHW{1,1});
    
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), layerName + ".bn2", eps);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        
    
    IConvolutionLayer* conv3 = network->addConvolutionNd(
        *relu2->getOutput(0), output_dim, DimsHW{1,1}, weightMap[layerName + ".conv3.weight"], emptywts
    );
//     std::cout<<output_dim<<std::endl;
    assert(conv3);
    conv3->setStrideNd(DimsHW{1,1});
    
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), layerName + ".bn3", eps);
    
    IElementWiseLayer* wise;
    // dowsample == 0: wise but stride = {1,1}
    // dowsample == 1: wise with stride ={2,2}
    // dowsample == -1: not wise
    
    if(dowsample>=0){
        IConvolutionLayer* conv4 = network->addConvolutionNd(
            input, output_dim, DimsHW{1,1}, weightMap[layerName + ".downsample.0.weight"], emptywts
        );
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride,stride});
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), layerName + ".downsample.1", eps);
        wise = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        wise = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    assert(wise);
    IActivationLayer* relu3 = network->addActivation(*wise->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3; 
}
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt){
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W}); // dynamic tensor shape
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights("../resnet50.wts");
//     for (auto z:weightMap){
//         std::cout<<z.size<<std::endl;
//     }
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});
    
    // bottneck1
    IActivationLayer* layer_1 = bottleneck(network,*pool1->getOutput(0), 64, 256, weightMap, "layer1.0",1e-5, 0);
    layer_1 = bottleneck(network, *layer_1->getOutput(0), 256, 256, weightMap, "layer1.1",1e-5, -1);
    layer_1 = bottleneck(network, *layer_1->getOutput(0), 256, 256, weightMap, "layer1.2",1e-5,-1);
    
    //bottneck2
    IActivationLayer* layer_2 = bottleneck(network,*layer_1->getOutput(0), 256, 512, weightMap, "layer2.0", 1e-5,1);
    layer_2 = bottleneck(network, *layer_2->getOutput(0), 512, 512, weightMap, "layer2.1",1e-5,-1);
    layer_2 = bottleneck(network, *layer_2->getOutput(0), 512, 512, weightMap, "layer2.2",1e-5,-1);
    layer_2 = bottleneck(network, *layer_2->getOutput(0), 512, 512, weightMap, "layer2.3",1e-5,-1);
  
    // bottneck3
    IActivationLayer* layer_3 = bottleneck(network,*layer_2->getOutput(0), 512, 1024, weightMap, "layer3.0", 1e-5,1);
    layer_3 = bottleneck(network,  *layer_3->getOutput(0), 1024, 1024, weightMap, "layer3.1",1e-5,-1);
    layer_3 = bottleneck(network,  *layer_3->getOutput(0), 1024, 1024, weightMap, "layer3.2",1e-5,-1);
    layer_3 = bottleneck(network, *layer_3->getOutput(0), 1024, 1024, weightMap, "layer3.3",1e-5,-1);
    layer_3 = bottleneck(network,  *layer_3->getOutput(0), 1024, 1024, weightMap, "layer3.4",1e-5,-1);
    layer_3 = bottleneck(network,  *layer_3->getOutput(0), 1024, 1024, weightMap, "layer3.5",1e-5,-1);
    
    // bottneck4
    
    IActivationLayer* layer_4 = bottleneck(network,*layer_3->getOutput(0), 1024, 2048, weightMap, "layer4.0", 1e-5,1);
    layer_4 = bottleneck(network,  *layer_4->getOutput(0), 2048, 2048, weightMap, "layer4.1",1e-5,-1);
    layer_4 = bottleneck(network,  *layer_4->getOutput(0), 2048, 2048, weightMap, "layer4.2",1e-5,-1);
    
    // fc
    
    IPoolingLayer* pool2 = network->addPoolingNd(*layer_4->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});
    
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), OUTPUT_SIZE, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);
    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));
    
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}
int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./resnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./resnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("resnet50.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("resnet50.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
    static const int size_data = 3 * INPUT_H * INPUT_W;
    float data[size_data];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static const int size_OUTPUT_SIZE = OUTPUT_SIZE;
    float prob[size_OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
