#include "yolov5_61.h"
#include "yolov5.h"
#include "yolo_onnx.h"


// yolov5_61.h
//int main()
//{
//	Net_config yolo_nets = { 0.3, 0.5, 0.3, "weights/yolov5s6.onnx" };
//	YOLO yolo_model(yolo_nets);
//	string imgpath = "images/bus.jpg";
//	Mat srcimg = imread(imgpath);
//	yolo_model.detect(srcimg);
//
//	static const string kWinName = "Deep learning object detection in OpenCV";
//	namedWindow(kWinName, WINDOW_NORMAL);
//	imshow(kWinName, srcimg);
//	waitKey(0);
//	destroyAllWindows();
//}


/*
// yolov5.h
int main(int argc, char **argv)
{

	std::vector<std::string> class_list = load_class_list();

	cv::Mat frame = cv::imread("images/zidane.jpg");

	bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

	cv::dnn::Net net;
	load_net(net, is_cuda);

	auto start = std::chrono::high_resolution_clock::now();
	int frame_count = 0;
	float fps = -1;
	int total_frames = 0;


	std::vector<Detection> output;
	detect(frame, net, output, class_list);


	int detections = output.size();

	for (int i = 0; i < detections; ++i)
	{

		auto detection = output[i];
		auto box = detection.box;
		auto classId = detection.class_id;
		const auto color = colors[classId % colors.size()];
		cv::rectangle(frame, box, color, 3);

		cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
		cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}


	cv::imshow("output", frame);
	cv::imwrite("bus_result.jpg", frame);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
*/



// yolov5_onnx.h
int main()
{
	Net_config yolo_nets = { 0.3, 0.5, 0.3,"weights/yolov5s6.onnx" };
	YOLO yolo_model(yolo_nets);
	string imgpath = "images/bus.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}

