/*
 * Classifier.h
 *
 *  Created on: Aug 31, 2015
 *      Author: Tzutalin
 */

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <vector>
#include <sstream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::vector<float> Prediction;

class Classifier {
    public:
        Classifier(const string& model_file,
                   const string& trained_file,
                   const string& mean_file,
                   const string& min_file,
                   const string& max_file);

        Prediction Classify(std::vector<float> input);

    private:
        void SetMean(const string& mean_file);

        void SetMin(const string& mean_file);
        
        void SetMax(const string& mean_file);

        std::vector<float> Predict(std::vector<float>);

        // void WrapInputLayer(std::vector<cv::Mat>* input_channels);

        void Preprocess(std::vector<float> &data);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        std::vector<float> mean_;
        std::vector<float> min_;
        std::vector<float> max_;
        // std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& min_file,
                       const string& max_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    std::cout << std::endl << "got initialized" << std::endl;
    net_.reset(new Net<float>(model_file, TEST));
    std::cout << std::endl << "got loaded" << std::endl;
    net_->CopyTrainedLayersFrom(trained_file);
    std::cout << std::endl << "got weighted" << std::endl;
    Blob<float>* input_layer = net_->input_blobs()[0];
    std::cout << std::endl << "got fetched blob" << std::endl;
    num_channels_ = input_layer->channels();
    std::cout << std::endl << "got channels: " << num_channels_ << std::endl;
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    std::cout << std::endl << "got dims: "
            << input_layer->width() << "," << input_layer->height() << std::endl;
    /* Load the binaryproto mean file. */
    SetMean(mean_file);
    SetMin(min_file);
    SetMax(max_file);
    std::cout << std::endl << "got meanminmax" << std::endl;
    Blob<float>* output_layer = net_->output_blobs()[0];
    std::cout << std::endl << "got outputs" << std::endl;
}

// static bool PairCompare(const std::pair<float, int>& lhs,
//                         const std::pair<float, int>& rhs) {
//     return lhs.first > rhs.first;
// }

/* Return the indices of the top N values of vector v. */
// static std::vector<int> Argmax(const std::vector<float>& v, int N) {
//     std::vector<std::pair<float, int> > pairs;
//     for (size_t i = 0; i < v.size(); ++i)
//         pairs.push_back(std::make_pair(v[i], i));
//     std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

//     std::vector<int> result;
//     for (int i = 0; i < N; ++i)
//         result.push_back(pairs[i].second);
//     return result;
// }

/* Return the top N predictions. */
Prediction Classifier::Classify(std::vector<float> input) {
    Prediction output = Predict(input);
    
    return output;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    std::cout << "data: " << blob_proto.data(0) << std::endl;
    std::cout << "data: " << blob_proto.data(1) << std::endl;
    std::cout << "data: " << blob_proto.data(2) << std::endl;
    std::cout << "data: " << blob_proto.data(3) << std::endl;
    std::cout << "data: " << blob_proto.data(4) << std::endl;
    std::cout << "data: " << blob_proto.data(5) << std::endl;
    std::cout << "data: " << blob_proto.data(6) << std::endl;
    std::cout << "data: " << blob_proto.data(7) << std::endl;
    
    Blob<float> mean_blob;
    std::cout << "yep";
    mean_blob.FromProto(blob_proto);
    std::cout << "bye";
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < mean_blob.channels(); i++) {
        mean_.push_back(*data);
        data += 1;
        std::cout << "something";
    }
    std::cout << "something";
}

void Classifier::SetMin(const string& min_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(min_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> min_blob;

    min_blob.FromProto(blob_proto);

    float* data = min_blob.mutable_cpu_data();
    for (int i = 0; i < min_blob.channels(); i++) {
        min_.push_back(*data);
        data += 1;
    }
}

void Classifier::SetMax(const string& max_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(max_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> max_blob;
    max_blob.FromProto(blob_proto);

    float* data = max_blob.mutable_cpu_data();
    for (int i = 0; i < max_blob.channels(); i++) {
        max_.push_back(*data);
        data += 1;
    }
}

/*
This will be the FORWARD PASS operation on input data
*/
std::vector<float> Classifier::Predict(std::vector<float> input) {
    std::cout << std::endl << "got here" << std::endl;
    Blob<float>* input_layer = net_->input_blobs()[0];
    std::cout << std::endl << "and here" << std::endl;
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    Preprocess(input);

    // @DEBUG: this probably won't work
    std::cout << "Input Num: " << input_layer->num()
            << "Input Channels: " << input_layer->channels()
            << "Input Width: " << input_layer->width()
            << "Input Height: " << input_layer->height()
            << std::endl;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); i++) {
        *input_data = input.at(i);
        input_data += (input_layer->width() * input_layer->height());
    }

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

/*
This will be where we normalize the data prior to the forward pass
*/
void Classifier::Preprocess(std::vector<float> &data) {
    // Normalize the data
    int desired_min = -1;
    int desired_max = 1;
    for (int i = 0; i < data.size(); i++) {
        data[i] = data[i] - mean_[i];
        data[i] = (((data[i] - min_[i]) * (desired_max - desired_min))/(max_[i] - min_[i])) 
            + desired_min;
    }
}

#endif
