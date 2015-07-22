#include "build/caffe.pb.h"
#include "3rdparty/sqlite3/sqlite3.h"

sqlite3* initDb(const char* filename);

void readProtoWithLayersSize(const caffe::NetParameter &netparam);

void writeConvnetParams( sqlite3 *db, int height, int width, int batchHeight, int batchWidth, int channels);
