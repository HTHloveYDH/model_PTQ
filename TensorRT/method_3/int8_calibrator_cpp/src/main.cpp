#include "int8_entropy_calibrator.h"

int main() {

    Logger logger;

    const char* calibrationImagesDir = "/home/cabin/h/calibration_images_5MP_depth/"; 
    const char* cacheFile = "/home/cabin/h/cachefiles/depth_vits_metric_int8_fp16_minmax_sim_c++_jetson.cache";
    const char* pathToOnnx = "/home/cabin/h/Quant_things/models/onnxsim_model/depth_anything_v2_metric_hypersim_vits_sim.onnx";
    const char* pathToEngine = "/home/cabin/h/engines/depth_vits_metric_int8_fp16_cache_minmax_sim_calib_on_jetson.trt";
    
    /* 
        create TensorRT builder and network
    */
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    initLibNvInferPlugins(&logger, "");
    /*
        for expicit batch:
    */
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(flag);
    
    /*
        for implicit batch:
    */
    // nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);


    /*
        parse ONNX model
    */
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(pathToOnnx, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));

    std::vector<int> sizeList = getTensorSizes(network);
    
    /* 
        create optimization profile
    */
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    //
    profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, network->getInput(0)->getDimensions().d[2], network->getInput(0)->getDimensions().d[3]});
    profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, network->getInput(0)->getDimensions().d[2], network->getInput(0)->getDimensions().d[3]});
    profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, network->getInput(0)->getDimensions().d[2], network->getInput(0)->getDimensions().d[3]});

    /*
        create builder config
    */
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->addOptimizationProfile(profile);

    //config->setBuilderOptimizationLevel(5);

    /* 
        Memory allocation
    */
    size_t workspace_size = (4ULL << 30);
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    config->setMaxWorkspaceSize(workspace_size);

    /* 
        comment this out if you don't need detailed information about the engine layers.
    */
    // config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    /* 
        comment this out if you want to create an engine with ONLY int8 precision.
    */
    // config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    /*
        Set INT8 mode and calibrator
    */
    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    // CrossRntropyCalib
    Int8EntropyCalibrator calibrator(
        10, // batch size for calibration 
        sizeList, // sizes of Dims
        network->getInput(0)->getDimensions().d[2], // input_w_
        network->getInput(0)->getDimensions().d[3], // input_h_
        calibrationImagesDir, 
        cacheFile, 
        network->getInput(0)->getName() 
    );
    config->setInt8Calibrator(&calibrator);


    getInfoOfLaunchedCommand(pathToOnnx, pathToEngine, cacheFile, profile, network, config);

    /* 
        Build engine
    */
    nvinfer1::IHostMemory *plan = builder->buildSerializedNetwork(*network, *config);
    
    std::ofstream engine_file(pathToEngine, std::ios::binary);
    if (!engine_file)
    {
        std::cout << "write serialized file failed\n";
    }
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    std::cout << "Engine build success!" << std::endl;

    /* 
        clean up
    */
    delete plan;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}
