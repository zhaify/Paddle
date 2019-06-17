set(INFERENCE_EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor benchmark)

if(WITH_GPU AND TENSORRT_FOUND)
    set(INFERENCE_EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps} ir_pass_manager analysis_predictor)
endif()

function(download_model install_dir model_name)
    if (NOT EXISTS ${install_dir})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL} ${model_name})
    endif()
endfunction()

function(download_model_and_data install_dir model_name data_name)
    if (NOT EXISTS ${install_dir})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL} ${model_name})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL} ${data_name})
    endif()
endfunction()

function(inference_analysis_api_test target install_dir filename)
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} benchmark
        ARGS --infer_model=${install_dir}/model --infer_data=${install_dir}/data.txt)
endfunction()

function(inference_analysis_api_int8_test target model_dir data_path filename)
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} benchmark
        ARGS --infer_model=${model_dir}/model
             --infer_data=${data_path}
             --warmup_batch_size=100
             --batch_size=1
             --paddle_num_threads=1
	     --iterations=1)
endfunction()
function(inference_analysis_api_test_with_fake_data target install_dir filename model_name disable_fc)
    download_model(${install_dir} ${model_name})
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${install_dir}/model
             --disable_mkldnn_fc=${disable_fc}) 
endfunction()

function(inference_analysis_api_test_with_refer_result target install_dir filename)
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${install_dir}/model --infer_data=${install_dir}/data.txt
             --refer_result=${install_dir}/result.txt)
endfunction()

# int8 image classification tests
if(WITH_MKLDNN)
  set(INT8_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")
  if (NOT EXISTS ${INT8_DATA_DIR})
    inference_download_and_uncompress(${INT8_DATA_DIR} "${INFERENCE_URL}/int8" "imagenet_val_100_tail.tar.gz")
  endif()
  
  if (NOT EXISTS ${INT8_DATA_DIR}/pascalvoc_data.bin)
    inference_download_and_uncompress(${INT8_DATA_DIR} "${INFERENCE_URL}/int8" "pascalvoc_val_200_head.tar.gz")
  endif()
  set(IMAGENET_DATA_PATH "${INT8_DATA_DIR}/data.bin")
  set(PASCALVOC_DATA_PATH "${INT8_DATA_DIR}/pascalvoc_data.bin")

  #mobilenet-ssd int8 model
   set(INT8_MOBILENET_SSD_MODEL_DIR "${INT8_DATA_DIR}/mobilenet-ssd")
   if (NOT EXISTS ${INT8_MOBILENET_SSD_MODEL_DIR})
     inference_download_and_uncompress(${INT8_MOBILENET_SSD_MODEL_DIR} "${INFERENCE_URL}/int8" "mobilenet_ssd_int8_model.tar.gz" )
   endif()
   inference_analysis_api_int8_test(test_analyzer_int8_mobilenet_ssd ${INT8_MOBILENET_SSD_MODEL_DIR} ${PASCALVOC_DATA_PATH} analyzer_int8_object_detection_tester.cc)

endif()
