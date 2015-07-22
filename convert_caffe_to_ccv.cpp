#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdexcept>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "build/caffe.pb.h"
#include "proto_reader.h"
#include "3rdparty/sqlite3/sqlite3.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::Message;

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  
    int fd = open(filename, O_RDONLY);
    if(fd < 0)
      return false;
    
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(1073741824, 536870912);
    
    bool success = proto->ParseFromCodedStream(coded_input);    
    delete coded_input;
    delete raw_input;
    close(fd);
    return success;
}

void readCaffeProto(caffe::NetParameter* netparam) {  
  
    if (netparam->layers_size() > 0)
       readProtoWithLayersSize(*netparam);
    else
      throw std::runtime_error("These kinds of models are not supported yet");
      //Not sure, I have seen in https://github.com/szagoruyko/loadcaffe/blob/master/loadcaffe.cpp
      //that such models can exist, but I have not encountered them yet.
 }
 

 

void loadCaffe() {  
  
    char const* prototxt_name = "bvlc_reference_caffenet/deploy.prototxt";
    char const* binary_name =   "bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    
    caffe::NetParameter* netparam = new caffe::NetParameter();
    
    //I'll try to extract the model architecture from the data structures inside
    //the binary model,without reading the deploy.prototxt file.
    bool success = ReadProtoFromBinaryFile(binary_name, netparam);
    if(success)
    {
      std::cout << "Successfully loaded " << binary_name << std::endl;
    }
    else
      std::cout << "Couldn't load " << binary_name << std::endl;    
      
    readCaffeProto(netparam);
    
}

int main()
{
  loadCaffe();
}



