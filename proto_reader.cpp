#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdexcept>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::Message;

#include "proto_reader.h"
#include "build/caffe.pb.h"
#include "3rdparty/sqlite3/sqlite3.h"

extern "C" {
	#include "3rdparty/libccv/ccv.h"
}

/**
* Create the .sqllite3 file and the empty tables
*/
sqlite3* initDb(const char* filename) {
    
	 
    sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char layer_create_table_qs[] =
			"CREATE TABLE IF NOT EXISTS layer_params "
			"(layer INTEGER PRIMARY KEY ASC, type INTEGER, "
			"input_matrix_rows INTEGER, input_matrix_cols INTEGER, input_matrix_channels INTEGER, input_matrix_partition INTEGER, input_node_count INTEGER, "
			"output_rows INTEGER, output_cols INTEGER, output_channels INTEGER, output_partition INTEGER, output_count INTEGER, output_strides INTEGER, output_border INTEGER, "
			"output_size INTEGER, output_kappa REAL, output_alpha REAL, output_beta REAL, output_relu INTEGER);"
			"CREATE TABLE IF NOT EXISTS convnet_params "
			"(convnet INTEGER PRIMARY KEY ASC, input_height INTEGER, input_width INTEGER, mean_activity BLOB);"
			"CREATE TABLE IF NOT EXISTS layer_data "
			"(layer INTEGER PRIMARY KEY ASC, weight BLOB, bias BLOB, half_precision INTEGER);";
			
		assert(SQLITE_OK == sqlite3_exec(db, layer_create_table_qs, 0, 0, 0));		
    }
    else
        assert(false);
    return db;    
}


/**
* Read from the object the params and the data
*/
void readProtoWithLayersSize(const caffe::NetParameter &netparam)
{

	//TODO: parameter for layer_data, understand when it is safe to put it = 1
	//For now I guess it is more safe = 0 (no half precision)
	int half_precision = 0;	
	

    sqlite3* db = initDb("miodb.sqllite3");
    
	const char layer_data_insert_qs[] =
			"REPLACE INTO layer_data "
			"(layer, weight, bias, half_precision) VALUES ($layer, $weight, $bias, $half_precision);";
	sqlite3_stmt* layer_data_insert_stmt = 0;		
	
	const char layer_params_insert_qs[] = 
	"REPLACE INTO layer_params "
	"(layer, type, "
	"input_matrix_rows, input_matrix_cols, input_matrix_channels, input_matrix_partition, input_node_count, "
	"output_rows, output_cols, output_channels, output_partition, output_count, output_strides, output_border, "
	"output_size, output_kappa, output_alpha, output_beta, output_relu) VALUES "
	"($layer, $type, " // 1
	"$input_matrix_rows, $input_matrix_cols, $input_matrix_channels, $input_matrix_partition, $input_node_count, " // 6
	"$output_rows, $output_cols, $output_channels, $output_partition, $output_count, $output_strides, $output_border, " // 13
	"$output_size, $output_kappa, $output_alpha, $output_beta, $output_relu);"; // 18
			
	sqlite3_stmt* layer_params_insert_stmt = 0;
	assert(SQLITE_OK == sqlite3_prepare_v2(db, layer_params_insert_qs, sizeof(layer_params_insert_qs), &layer_params_insert_stmt, 0));
	assert(SQLITE_OK == sqlite3_prepare_v2(db, layer_data_insert_qs, sizeof(layer_data_insert_qs), &layer_data_insert_stmt, 0));


  int num_output = netparam.input_dim_size();  
  int i_ccv = 0;
  
  int nextInputRows;
  int nextInputCols;
  int nextInputChannels;
  int nextInputPartitions;
  int batchSize; //square side
  
  
  int inputWidth = -1;
  int inputHeight = -1;
  int inputChannels = -1;
  bool inputSaved = false;
  
  
   /**
    * Iterate layers.
	* For each layer I have to fill 1 record for each
	* of the  2 tables:
	* -  layer_data (insert the weights of the layer)
	* -  layer_params (insert the configuration parameters of the layer)
	*/
  for (int i=0; i<netparam.layers_size(); ++i)
  {
    std::vector<std::pair<std::string, std::string>> lines;
    auto& layer = netparam.layers(i);
	
	//Note: I use braces {} on case blocks because I have variables with the same name	
    switch(layer.type())
    {
		
		//these 3 layers do not need sqllite representation
		case caffe::V1LayerParameter::RELU: //no params      
		case caffe::V1LayerParameter::SOFTMAX_LOSS:     //no params
		case caffe::V1LayerParameter::SOFTMAX: //no params
		case caffe::V1LayerParameter::DROPOUT:  
		//has a param: layer.dropout_param().dropout_ratio() ,
		//but it should be used only during training not during test phase  
		break;		  
					
		case caffe::V1LayerParameter::DATA: //first layer, input
		{
			auto &paramD = layer.data_param();
			int crop_size = paramD.crop_size();
			batchSize = paramD.batch_size();
			nextInputRows = crop_size;
			nextInputCols = crop_size;	  	  
			break;	  		  		
		} 	
		case caffe::V1LayerParameter::CONVOLUTION:      	
		{  
			i_ccv += 1;
			auto &param = layer.convolution_param();
			int groups = param.group() == 0 ? 1 : param.group();
			int nInputPlane = layer.blobs(0).channels()*groups;
			
			int channels = layer.blobs(0).channels();
				
			//TODO Empirically set at 2 looking at the existing models. 
			//It should be good with Mattnet. What about AlexNet? And other nets?	
			int outputPartitions = 2; 
			
			int nOutputPlane = layer.blobs(0).num();
			num_output = nOutputPlane;  //TODO merge num_ouput and nOutputPlane
			int kW = param.kernel_w();
			int kH = param.kernel_h();
			int dW = param.stride_w();
			int dH = param.stride_h();
			if(kW==0 || kH==0)
			{
			  kW = param.kernel_size();
			  kH = kW;
			}
			if(dW==0 || dH==0)
			{
			  dW = param.stride();
			  dH = dW;
			}
			int pad_w = param.pad_w();
			int pad_h = param.pad_h();
			if(pad_w==0 || pad_h==0)
			{
			  pad_w = param.pad();
			  pad_h = pad_w;
			}
			
			//first convolution layer, store the params width,height,channel
			if (!inputSaved)  
			{
				inputWidth = nextInputCols;
				inputHeight = nextInputRows;
				inputChannels = channels;
				inputSaved = true; 
			}
			
			
			
			/** WRITE weights on LAYER_DATA **/
			
			caffe::BlobProto blobs0 = layer.blobs(0);
			//RepeatedField is kind of an optimized Vector
			::google::protobuf::RepeatedField< float > weightsRf = layer.blobs(0).data();
			float* weightsUnderlyingPointer = weightsRf.mutable_data();
			::google::protobuf::RepeatedField< float > biasRf = layer.blobs(1).data();
			float* biasUnderlyingPointer = biasRf.mutable_data();
			
			int wnum = kW* kH * channels * num_output / groups;
			
			sqlite3_bind_int(layer_data_insert_stmt, 1, i_ccv);
			if (half_precision == 1)
			{
				uint16_t* w = (uint16_t*)ccmalloc(sizeof(uint16_t) * wnum);
				ccv_float_to_half_precision(weightsUnderlyingPointer, w, wnum);
				uint16_t* bias = (uint16_t*)ccmalloc(sizeof(uint16_t) * num_output);
				ccv_float_to_half_precision(biasUnderlyingPointer, bias,	num_output);
				sqlite3_bind_blob(layer_data_insert_stmt, 2, w, sizeof(uint16_t) * wnum, ccfree);
				sqlite3_bind_blob(layer_data_insert_stmt, 3, bias, sizeof(uint16_t) * num_output, ccfree);
			} else {
				sqlite3_bind_blob(layer_data_insert_stmt, 2, weightsUnderlyingPointer, sizeof(float) * wnum, SQLITE_STATIC);
				sqlite3_bind_blob(layer_data_insert_stmt, 3, biasUnderlyingPointer, sizeof(float) * num_output, SQLITE_STATIC);
			}
			sqlite3_bind_int(layer_data_insert_stmt, 4, half_precision);
			int resultOfOperation = sqlite3_step(layer_data_insert_stmt);
						
			if (resultOfOperation != SQLITE_DONE)
			{
				const char* errmsg = sqlite3_errmsg(db);
				printf("ERR MSG:%s",errmsg);
			}
			assert(SQLITE_DONE == resultOfOperation);
			sqlite3_reset(layer_data_insert_stmt);
			sqlite3_clear_bindings(layer_data_insert_stmt);
			/** END OF WRITING weights on LAYER_DATA */
			
			
			//write Convolutional layer in sqllite [LAYER_PARAMS]
			sqlite3_bind_int(layer_params_insert_stmt, 1, i_ccv);
			sqlite3_bind_int(layer_params_insert_stmt, 2, CCV_CONVNET_CONVOLUTIONAL);
			sqlite3_bind_int(layer_params_insert_stmt, 3, nextInputRows); //layer->input.matrix.rows
			sqlite3_bind_int(layer_params_insert_stmt, 4, nextInputCols); //layer->input.matrix.cols
			sqlite3_bind_int(layer_params_insert_stmt, 5, nInputPlane); //layer->input.matrix.channels 
			sqlite3_bind_int(layer_params_insert_stmt, 6, groups); //input matrix partitions
			sqlite3_bind_int(layer_params_insert_stmt, 7, 0); //layer->input.node.count [Only for dense layers]
			sqlite3_bind_int(layer_params_insert_stmt, 8, kH); //rows = height [kernel]
			sqlite3_bind_int(layer_params_insert_stmt, 9, kW); //cols = width [kernel]
			sqlite3_bind_int(layer_params_insert_stmt, 10, channels); //layer->net.convolutional.channels 
			sqlite3_bind_int(layer_params_insert_stmt, 11, outputPartitions); //output partitions
			sqlite3_bind_int(layer_params_insert_stmt, 12, num_output); //layer->net.convolutional.count
			sqlite3_bind_int(layer_params_insert_stmt, 13, dW); //stride = stride [only square on ccv]
			sqlite3_bind_int(layer_params_insert_stmt, 14, pad_w); //border = pad [only square on ccv] | to be confirmed
					
			assert(SQLITE_DONE == sqlite3_step(layer_params_insert_stmt));
			sqlite3_reset(layer_params_insert_stmt);
			sqlite3_clear_bindings(layer_params_insert_stmt);
			
			
			int nextInputTmp = ((nextInputRows - kW + 2 * pad_w) / dW ) + 1;
			nextInputRows = nextInputTmp;
			nextInputCols = nextInputTmp; 
			nextInputChannels = num_output;
			nextInputPartitions = outputPartitions; 
			  
			break;      
			}	  
      	case caffe::V1LayerParameter::POOLING:
     	{
			i_ccv += 1;
			auto &param = layer.pooling_param();
			int poolType = param.pool() == caffe::PoolingParameter::MAX ? CCV_CONVNET_MAX_POOL : CCV_CONVNET_AVERAGE_POOL;
			
			int kW = param.kernel_w();
			int kH = param.kernel_h();
			int dW = param.stride_w();
			int dH = param.stride_h();
			if(kW==0 || kH==0)
			{
			  kW = param.kernel_size();
			  kH = kW;
			}
			if(dW==0 || dH==0)
			{
			  dW = param.stride();
			  dH = dW;
			}
			int pad_w = param.pad_w();
			int pad_h = param.pad_h();
			if(pad_w==0 || pad_h==0)
			{
			  pad_w = param.pad();
			  pad_h = pad_w;
			}
			
			sqlite3_bind_int(layer_params_insert_stmt, 1, i_ccv);
			sqlite3_bind_int(layer_params_insert_stmt, 2, poolType);
			sqlite3_bind_int(layer_params_insert_stmt, 3, nextInputRows); //layer->input.matrix.rows
			sqlite3_bind_int(layer_params_insert_stmt, 4, nextInputCols); //layer->input.matrix.cols
			sqlite3_bind_int(layer_params_insert_stmt, 5, nextInputChannels); //layer->input.matrix.channels 
			sqlite3_bind_int(layer_params_insert_stmt, 6, nextInputPartitions); //input matrix partitions
			sqlite3_bind_int(layer_params_insert_stmt, 7, 0); //layer->input.node.count [Only for dense layers]
			sqlite3_bind_int(layer_params_insert_stmt, 13, dW); //stride = stride [only square on ccv]
			sqlite3_bind_int(layer_params_insert_stmt, 14, pad_w); //border = pad [only square on ccv] | to be confirmed
			sqlite3_bind_int(layer_params_insert_stmt, 15, kW); //layer->net.pool.size
			
			assert(SQLITE_DONE == sqlite3_step(layer_params_insert_stmt));
			sqlite3_reset(layer_params_insert_stmt);
			sqlite3_clear_bindings(layer_params_insert_stmt);
			
			int nextInputTmp = ((nextInputRows - kW + 2 * pad_w) / dW ) + 1;
			nextInputRows = nextInputTmp;
			nextInputCols = nextInputTmp;
			      
			break;
	 	}
		case caffe::V1LayerParameter::LRN: //no layer_data, only layer_params
		{
			i_ccv += 1;
			auto &param = layer.lrn_param();
			int local_size = param.local_size();
			float alpha = param.alpha();
			float beta = param.beta();
			float k = param.k();
			
			sqlite3_bind_int(layer_params_insert_stmt, 1, i_ccv);
			sqlite3_bind_int(layer_params_insert_stmt, 2, CCV_CONVNET_LOCAL_RESPONSE_NORM);
			sqlite3_bind_int(layer_params_insert_stmt, 3, nextInputRows); //layer->input.matrix.rows
			sqlite3_bind_int(layer_params_insert_stmt, 4, nextInputCols); //layer->input.matrix.cols
			sqlite3_bind_int(layer_params_insert_stmt, 5, nextInputChannels); //layer->input.matrix.channels 
			sqlite3_bind_int(layer_params_insert_stmt, 6, nextInputPartitions); //input matrix partitions
			sqlite3_bind_int(layer_params_insert_stmt, 7, 0); //layer->input.node.count [Only for dense layers]
			//params only for local response normalization layer:
			sqlite3_bind_int(layer_params_insert_stmt, 15, local_size); //layer->net.rnorm.size
			sqlite3_bind_double(layer_params_insert_stmt, 16, k); // layer->net.rnorm.kappa
			sqlite3_bind_double(layer_params_insert_stmt, 17, alpha); //layer->net.rnorm.alpha
			sqlite3_bind_double(layer_params_insert_stmt, 18, beta); //layer->net.rnorm.beta
			
			
			assert(SQLITE_DONE == sqlite3_step(layer_params_insert_stmt));
			sqlite3_reset(layer_params_insert_stmt);
			sqlite3_clear_bindings(layer_params_insert_stmt);
			
			break;
		}
		case caffe::V1LayerParameter::INNER_PRODUCT: //full connect layer
      	{
			i_ccv += 1;
	        auto &param = layer.inner_product_param();
	        int nInputPlane = layer.blobs(0).width();
	        int nOutputPlane = param.num_output();
	
		
			/** WRITE weights on LAYER_DATA **/
		
			//from ccv_convnet.c :layers[i].wnum = params[i].input.node.count * params[i].output.full_connect.count
			int wnum = nInputPlane * nOutputPlane;	;
			//RepeatedField is kind of an optimized Vector
			::google::protobuf::RepeatedField< float > weightsRf = layer.blobs(0).data();
			float* weightsUnderlyingPointer = weightsRf.mutable_data();
			::google::protobuf::RepeatedField< float > biasRf = layer.blobs(1).data();
			float* biasUnderlyingPointer = biasRf.mutable_data();
		
	
	
			sqlite3_bind_int(layer_data_insert_stmt, 1, i_ccv);
			if (half_precision == 1)
			{
				uint16_t* w = (uint16_t*)ccmalloc(sizeof(uint16_t) * wnum);
				ccv_float_to_half_precision(weightsUnderlyingPointer, w, wnum);
				uint16_t* bias = (uint16_t*)ccmalloc(sizeof(uint16_t) * nOutputPlane);
				ccv_float_to_half_precision(biasUnderlyingPointer, bias, nOutputPlane);
				sqlite3_bind_blob(layer_data_insert_stmt, 2, w, sizeof(uint16_t) * wnum, ccfree);
				sqlite3_bind_blob(layer_data_insert_stmt, 3, bias, sizeof(uint16_t) * nOutputPlane, ccfree);
			} else {
				sqlite3_bind_blob(layer_data_insert_stmt, 2, weightsUnderlyingPointer, sizeof(float) * wnum, SQLITE_STATIC);
				sqlite3_bind_blob(layer_data_insert_stmt, 3, biasUnderlyingPointer, sizeof(float) * nOutputPlane, SQLITE_STATIC);
			}
			sqlite3_bind_int(layer_data_insert_stmt, 4, half_precision);
			assert(SQLITE_DONE == sqlite3_step(layer_data_insert_stmt));
			sqlite3_reset(layer_data_insert_stmt);
			sqlite3_clear_bindings(layer_data_insert_stmt);
		/** END of WRITE weights on LAYER_DATA */
	
		/** WRITE LAYER_PARAMS **/
			
			int followedByRelu = 0; //false
			if (i  < netparam.layers_size() - 1)
				{
					auto& nextLayer = netparam.layers(i + 1);
					if (nextLayer.type() == caffe::V1LayerParameter::RELU )
						followedByRelu = 1;  //true, relu on next layer
					
				}
			sqlite3_bind_int(layer_params_insert_stmt, 1, i_ccv);
			sqlite3_bind_int(layer_params_insert_stmt, 2, CCV_CONVNET_FULL_CONNECT);
			sqlite3_bind_int(layer_params_insert_stmt, 3, nextInputRows); //layer->input.matrix.rows
			sqlite3_bind_int(layer_params_insert_stmt, 4, nextInputCols); //layer->input.matrix.cols
			sqlite3_bind_int(layer_params_insert_stmt, 5, nextInputChannels); //layer->input.matrix.channels		
			int inputPartitions = 1; //TODO confirm it's right, on the sample nets it is always 1.
			sqlite3_bind_int(layer_params_insert_stmt, 6, inputPartitions); //input matrix partitions		
			//nInputPlane is  nextInputCols * nextInputRows * nextInputChannels * inputPartitions;
			sqlite3_bind_int(layer_params_insert_stmt, 7, nInputPlane); //layer->input.node.count [Only for dense layers]				
			sqlite3_bind_int(layer_params_insert_stmt, 12, nOutputPlane); //layer->net.full_connect.count
			sqlite3_bind_int(layer_params_insert_stmt, 19, followedByRelu); //layer->net.full_connect.relu		
			assert(SQLITE_DONE == sqlite3_step(layer_params_insert_stmt));
			sqlite3_reset(layer_params_insert_stmt);
			sqlite3_clear_bindings(layer_params_insert_stmt);
			
			nextInputRows = nOutputPlane; //TODO not sure
			nextInputCols = 1; //TODO coherent with the existing model, but I am not sure it's right
			nextInputChannels = 1; //TODO coherent with the existing model, but I am not sure it's right
			/** END LAYER_PARAMS*/
	        break;
		}
     	default:
		{      
			std::cout << "MODULE " << i << " " << layer.name() << " UNDEFINED\n";
			break;
		}      
    } //end of switch on layer type
   } //end of layer iteration

 	writeConvnetParams(db,inputHeight, inputWidth, batchSize, batchSize, inputChannels);	

	sqlite3_finalize(layer_data_insert_stmt);	
	sqlite3_finalize(layer_params_insert_stmt);
		
	sqlite3_close(db);
	
	}



/**
* Mean image depends on the dataset used for training, not on the model.
* example: caffe/data/ilsvrc12/imagenet_mean.binaryproto
*
* FIXME: I have encountered issues including the C++ caffe parts,
* we can try to fix the includes or execute this step by itself,
* on a c++ executable that includes only caffe libraries and some sqlite interface 
*
*/
/*
float* getMeanImage(char* filename, int num_channels_) {
	
	//step 1: read from the file a protobuf structure
	caffe::BlobProto proto;
  	ReadProtoFromBinaryFileOrDie(filename, &proto);
	    
	//step2, convert BlobProto in Blob<float>
	Blob<float> mean_blob;
  	mean_blob.FromProto(blob_proto);

	//step 3: return array of floats
	float* data = proto.mutable_cpu_data();
	return data;
}*/

/**
* Stub method to get workflow running
*/
float* getMeanImage(int height, int width, int channels)
{
	int type = channels | CCV_32F;
	ccv_dense_matrix_t* mean_image_empty_matrix = ccv_dense_matrix_new(height, width, type, 0, 0);
	return mean_image_empty_matrix->data.f32;
}

/**
* Write the record inside the convnet_param table
*/
void writeConvnetParams( sqlite3 *db, int height, int width, int batchHeight, int batchWidth, int channels) {
	
	
	  // insert convnet related params
		const char convnet_params_insert_qs[] =
			"REPLACE INTO convnet_params "
			"(convnet, mean_activity, input_height, input_width) VALUES (0, $mean_activity, $input_height, $input_width);";
		sqlite3_stmt* convnet_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, convnet_params_insert_qs, sizeof(convnet_params_insert_qs), &convnet_params_insert_stmt, 0));
				
		uint64_t sixth_param = 0;
		
		
		float* mean = getMeanImage(height,width,channels);;
		
		
		sqlite3_bind_blob(convnet_params_insert_stmt, 1, mean , sizeof(float) * height * width * channels, SQLITE_STATIC);
		sqlite3_bind_int(convnet_params_insert_stmt, 2, batchHeight);
		sqlite3_bind_int(convnet_params_insert_stmt, 3, batchWidth);
		assert(SQLITE_DONE == sqlite3_step(convnet_params_insert_stmt));
		
		sqlite3_reset(convnet_params_insert_stmt);
		sqlite3_clear_bindings(convnet_params_insert_stmt);
		
		sqlite3_finalize(convnet_params_insert_stmt);
		
}
