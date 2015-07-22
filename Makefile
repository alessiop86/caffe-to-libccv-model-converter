CXX = g++ $(CXXFLAGS) -std=c++11
CC = gcc $(CFLAGS)

all:	convertCaffeTarget

debug: CXXFLAGS += -DDEBUG -g 
debug: CFLAGS += -DDEBUG -g 
debug: convertCaffeTarget

clean:
	rm -f miodb.sqllite3 *.o  	

convertCaffeTarget:	libccv  3rdparty/sqlite3/sqlite3.o convertCaffe

libccv:
	 $(CC) 3rdparty/libccv/ccv_util.c 3rdparty/libccv/ccv_memory.c 3rdparty/libccv/ccv_cache.c 3rdparty/libccv/sha1/sha1.c -c   	

3rdparty/sqlite3/sqlite3.o: 3rdparty/sqlite3/sqlite3.c
	$(CC) $< -o $@ -c -O3 -D SQLITE_THREADSAFE=0 -D SQLITE_OMIT_LOAD_EXTENSION 

convertCaffe:
	$(CXX) proto_reader.cpp convert_caffe_to_ccv.cpp build/caffe.pb.cc  -o convertCaffe.out -lprotobuf 3rdparty/sqlite3/sqlite3.o ccv_memory.o ccv_cache.o  ccv_util.o sha1.o

#END


#attempt failed at compiling an object filed and include it in loadCaffe in order to use
#caffe data structures in loadCaffe (for example caffe::Blob<float> 
#caffeLight2.o:
#	 $(CXX) -fpermissive 3rdparty/caffe/blob.cpp -o $@ -c
#cleanCaffeLight:
#	rm -f 3rdparty/caffe/*.gch  3rdparty/caffe/*.hpp~  3rdparty/caffe/util/*.gch  3rdparty/caffe/util/*.hpp~