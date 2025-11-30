#include "RecoE2E/FrameProducers/interface/predict_tf.h"

e2e::Frame2D e2e::predict_tf(e2e::Frame4D& vinputFrame, string model_filename, string input_layer_name, string output_layer_name){
 e2e::Frame2D output_preds;
 tensorflow::Session* session;
 tensorflow::GraphDef graph_def;
 tensorflow::SessionOptions opts;
 std::vector<tensorflow::Tensor> outputs; // Store outputs
 // create a new session
 TF_CHECK_OK(NewSession(opts, &session));
 
 std::string graph_definition="RecoE2E/"+model_filename;
 //std::cout<<" >> Running Inference."<<endl;
 edm::LogInfo("JetFrameProducer") << " >> Running Inference.";
 int batch_sz = vinputFrame.size();
 if (batch_sz>0){ 
  int no_channels = vinputFrame[0].size();
  //std::cout<<no_channels<<std::endl;
  edm::LogInfo("JetFrameProducer") << no_channels;
  if (no_channels>0){
   int frame_height = vinputFrame[0][0].size();
   if (frame_height>0){
    int frame_width = vinputFrame[0][0][0].size();
    if (frame_width>0){
     //TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));
     // load the graph definition, i.e. an object that contains the computational graph
     tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(graph_definition);
     tensorflow::Tensor tmp(tensorflow::DT_FLOAT, tensorflow::TensorShape({frame_height, frame_width}));
  
     tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_sz, frame_height, frame_width,  no_channels}));
     auto _XTensor = x.tensor<float,4>();
     for (int batch_idx=0; batch_idx<batch_sz; batch_idx++){
      for (int row_idx=0;row_idx<frame_height; row_idx++){
       for (int col_idx=0; col_idx<frame_width; col_idx++){
        for (int depth_idx=0; depth_idx<no_channels; depth_idx++){
         _XTensor(batch_idx, row_idx, col_idx, depth_idx) = vinputFrame[batch_idx][depth_idx][row_idx][col_idx];
        }
       }
      }
     }
     /*if(!x.CopyFrom(tmp, tensorflow::TensorShape({1, frame_height, frame_width, 1}))){
      std::cout<<" >> Reshape not successfull."<<endl;
     }*/
     // Set GPU options
     tensorflow::graph::SetDefaultDevice("/gpu:0", &graph_def);
     opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
     opts.config.mutable_gpu_options()->set_allow_growth(true);
 
     //int GPUID = std::stoi(params->getGpuDeviceStr());
     int GPUID = 0;
     setenv("CUDA_VISIBLE_DEVICES", "", GPUID);

     //std::cout << "Initial  visible_device_list : "<<opts.config.gpu_options().visible_device_list() << std::endl;
     opts.config.mutable_gpu_options()->set_allow_growth(true);
     opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);//params->getGpuMemoryRatio());
 
 
     // Load graph into session
     //TF_CHECK_OK(session->Create(graph_def));
 
     // create a session
     session = tensorflow::createSession(graphDef);
 
     // Initialize our variables
 
     TF_CHECK_OK(session->Run({{input_layer_name/*"inputs"*/, x}/*, {"y", y}*/}, {output_layer_name/*"softmax_1/Sigmoid"*/}, {}, &outputs)); // Get output
     //tensorflow::run(session, { { "x", x }, {"y", y} }, { "cost" }, &outputs);
     //std::cout<<" >> Classification predictions: "<<std::endl;
     edm::LogInfo("JetFrameProducer") << " >> Classification predictions: ";     

     if (outputs[0].shape().dims()!=2) edm::LogInfo("JetFrameProducer") << " * Expected 2 dimensional output. Received " << outputs[0].shape().dims() << " dimensional output."; //std::cout<<"* Expected 2 dimensional output. Received "<<outputs[0].shape().dims()<<" dimension output."<<std::endl;
     else {
      e2e::Frame1D preds;
      for (int row_idx=0; row_idx<outputs[0].shape().dim_size(0); row_idx++){
       for (int col_idx=0; col_idx<outputs[0].shape().dim_size(1); col_idx++){
        preds.push_back( outputs[0].matrix<float>()(row_idx,col_idx) );
       }
       output_preds.push_back(preds);
      }
      for (int row_idx=0; row_idx<outputs[0].shape().dim_size(0); row_idx++){
       for (int col_idx=0; col_idx<outputs[0].shape().dim_size(1); col_idx++){
        //std::cout<<"outputs: ("<<row_idx<<","<<col_idx<<"): "<<output_preds[row_idx][col_idx]<<std::endl;
	edm::LogInfo("JetFrameProducer") << "outputs: (" << row_idx << "," << col_idx << "): " << output_preds[row_idx][col_idx];
       }
      }
     }
     //std::cout<<" >> Batch size is: "<<outputs[0].shape().dim_size(0)<<std::endl;
     edm::LogInfo("JetFrameProducer") << " >> Batch size is: " << outputs[0].shape().dim_size(0);
     outputs.clear();
  
     session->Close();
     delete session;
     //std::cout<<" >> Classification done"<<endl;
     edm::LogInfo("JetFrameProducer") << " >> Classification done.";
    }
    else{
     //std::cout<<"* Shape Error: Invalid Width(<0) dimension. Expected format: (N, C, H, W)"<<std::endl;
     edm::LogInfo("JetFrameProducer") << " * Shape Error: Invalid Width (<0) dimension. Expected format: (N, C, H, W)";	
    }
   }
   else{
    //std::cout<<"* Shape Error: Invalid Height(<0) dimension. Expected format: (N, C, H, W)"<<std::endl;
    edm::LogInfo("JetFrameProducer") << " * Shape Error: Invalid Height (<0) dimension. Expected format: (N, C, H, W)";
   }
  }
  else{
   //std::cout<<"* Shape Error: Invalid Channel(<0) dimension. Expected format: (N, C, H, W)"<<std::endl;
   edm::LogInfo("JetFrameProducer") << " * Shape Error: Invalid Channel (<0) dimension. Expected format: (N, C, H, W)";
  }
 }
 else{
  //std::cout<<"* Shape Error: Invalid Batch(<0) dimension. Expected format: (N, C, H, W)"<<std::endl;
  edm::LogInfo("JetFrameProducer") << " * Shape Error: Invalid Batch (<0) dimension. Expected format: (N, C, H, W)";
 }
 // cleanup
 //tensorflow::closeSession(session);
 //delete graphDef;
 return output_preds;
}
