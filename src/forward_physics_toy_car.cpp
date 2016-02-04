/*
 * ros_caffe_test.cpp
 *
 *  Created on: Aug 31, 2015
 *      Author: Tzutalin
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "Classifier_toycar.h"
#include <std_msgs/Float32MultiArray.h>

const std::string RECEIVE_IMG_TOPIC_NAME = "/toy_car_input";
const std::string PUBLISH_RET_TOPIC_NAME = "/toy_car_output";

Classifier* classifier;
std::string model_path;
std::string weights_path;
std::string mean_file;
std::string min_file;
std::string max_file;

ros::Publisher gPublisher;

void publishRet(Prediction& prediction);

void inputCallback(const std_msgs::Float32MultiArray::ConstPtr& input) {
    float data[8];
    for (int i = 0; i < (sizeof(data) / sizeof(data[0])); i++) {
        data[i] = input->data[i];
    }

    std::vector<float> input_data(data, data + (sizeof(data) / sizeof(data[0])));

    Prediction prediction = classifier->Classify(input_data);
	
    publishRet(prediction);
}

// TODO: Define a msg or create a service
// Try to receive : $rostopic echo /caffe_ret
void publishRet(Prediction& prediction)  {
    std_msgs::Float32MultiArray msg;
    for (int i = 0; i < prediction.size(); i++) {
      msg.data.push_back(prediction[i]);
    }
    gPublisher.publish(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_caffe_test");
    ros::NodeHandle nh;
    // To receive an image from the topic, PUBLISH_RET_TOPIC_NAME
    ros::Subscriber sub = nh.subscribe(RECEIVE_IMG_TOPIC_NAME, 1, inputCallback);
	gPublisher = nh.advertise<std_msgs::Float32MultiArray>(PUBLISH_RET_TOPIC_NAME, 100);
    const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = ROOT_SAMPLE + "/caffe/examples/toy_car/PhysicsLearner/toycar_2fc_deploy.prototxt";
    weights_path = ROOT_SAMPLE + "/caffe/examples/toy_car/PhysicsLearner/2fc_iter_20001.caffemodel";
    mean_file = ROOT_SAMPLE + "/caffe/examples/toy_car/PhysicsLearner/toy_car_mean.binaryproto";
    min_file = ROOT_SAMPLE + "/caffe/examples/toy_car/PhysicsLearner/toy_car_min.binaryproto";
    max_file = ROOT_SAMPLE + "/caffe/examples/toy_car/PhysicsLearner/toy_car_max.binaryproto";
    // label_file = ROOT_SAMPLE + "/data/synset_words.txt";
    // image_path = ROOT_SAMPLE + "/data/cat.jpg";

    classifier = new Classifier(model_path, weights_path, mean_file, min_file, max_file);

    // Test data/cat.jpg
    // cv::Mat img = cv::imread(image_path, -1);

 //    std::vector<Prediction> predictions = classifier->Classify(img);
 //    /* Print the top N predictions. */
 //    std::cout << "Test default image under /data/cat.jpg" << std::endl;
 //    for (size_t i = 0; i < predictions.size(); ++i) {
 //        Prediction p = predictions[i];
 //        std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
 //    }
	// publishRet(predictions);

    ros::spin();
    delete classifier;
    ros::shutdown();
    return 0;
}
