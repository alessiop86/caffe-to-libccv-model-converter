# caffe-to-libccv-model-converter

Unfinished C++ tool to convert [caffe](https://github.com/BVLC/caffe) models in [ccv](https://github.com/liuliu/ccv) models.I have discontinued this project, so contributions are highly appreciated and encouraged. 

I apologize in advance: **the code quality is very poor** (I am not a C nor C++ programmer, I wrote this code on late nights overlapping reading functions from caffe to writing functions from ccv: 2 complex codebases in 2 different languages, and I am very little familiar with the four of them).

Compile with "make all" and run with "./convertCaffe.out" to attempt the conversion of the caffe model available in the bvlc_reference_caffenet folder. The model is a variation of [AlexNet architecture](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) by Jeff Donahue, available for download with the caffe installation.

I managed to convert all the weights and the parameters of the layers of the architecture, but I still had 2 unresolved problems:
1 - Conversion of the mean image from caffe to ccv (I needed a missing dependency from caffe in order to extract the content and write in the ccv model in form of an array of floats).
2 - I skipping the mean image by replacing it with an empty matrix of the same dimension, and went ahead. Unfortunately when i tried to classify an image using the newly converted model, I encountered some dimension errors. My first (unconfirmed) hypothesis related those errors to the variation from AlexNet original architecture: ccv includes some assert to verify that the architecture of the CCN you are using for classification is somewhat coherent with the kind of models it expects to receive.
