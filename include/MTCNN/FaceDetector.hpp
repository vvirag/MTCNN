/*
 * FaceDetector.hpp
 *
 *  Created on: Mar 30, 2018
 *      Author: vvirag
 */

#ifndef INCLUDE_MTCNN_FACEDETECTOR_HPP_
#define INCLUDE_MTCNN_FACEDETECTOR_HPP_

#include <vector>
#include <string>

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/shared_ptr.hpp>

namespace mtcnn
{
	class FaceDetector
	{
	public:
		struct FaceInfo
		{
		};

		enum COLOR_ORDER {
			GRAY,
			RGBA,
			RGB,
			BGRA,
			BGR
		};

		enum MODEL_VERSION {
			MODEL_V1,
			MODEL_V2
		};

		enum NMS_TYPE {
			MIN,
			UNION,
		};

		enum IMAGE_DIRECTION {
			ORIENT_LEFT,
			ORIENT_RIGHT,
			ORIENT_UP,
			ORIENT_DOWN,
		};

		struct BoundingBox {
			//rect two points
			float x1, y1;
			float x2, y2;
			//regression
			float dx1, dy1;
			float dx2, dy2;
			//cls
			float score;
			//inner points
			float points_x[5];
			float points_y[5];
		};

		struct CmpBoundingBox
		{
			bool operator() (const BoundingBox& b1, const BoundingBox& b2)
			{
				return b1.score > b2.score;
			}
		};

		FaceDetector(const std::string& model_dir,
					 const MODEL_VERSION model_version);

		std::vector< BoundingBox > Detect (const cv::Mat& img, const COLOR_ORDER color_order, const IMAGE_DIRECTION orient, int min_size = 20, float P_thres = 0.6, float R_thres = 0.7, float O_thres =0.7, bool is_fast_resize = true, float scale_factor = 0.709);

		cv::Size GetInputSize()   { return input_geometry_; }
		int      GetInputChannel(){ return num_channels_; }
		std::vector<int> GetInputShape()  {
			caffe::Blob<float>* input_layer = P_Net->input_blobs()[0];
			return input_layer->shape();
		}

	private:
		void generateBoundingBox(const std::vector<float>& boxRegs, const std::vector<int>& box_shape,
								 const std::vector<float>& cls, const std::vector<int>& cls_shape,
								 float scale, float threshold, std::vector<BoundingBox>& filterOutBoxes);

		void filteroutBoundingBox(const std::vector<BoundingBox>& boxes,
								  const std::vector<float>& boxRegs, const std::vector<int>& box_shape,
								  const std::vector<float>& cls, const std::vector<int>& cls_shape,
								  const std::vector< float >& points, const std::vector< int >& points_shape,
								  float threshold, std::vector<BoundingBox>& filterOutBoxes);

		void nms_cpu(std::vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, std::vector<BoundingBox>& filterOutBoxes);

		//void pad(vector<BoundingBox>& boxes, int imgW, int imgH);

		//vector<int> nms(vector<int> boxs, );
		std::vector<float> predict(const cv::Mat& img);
		void wrapInputLayer(boost::shared_ptr< caffe::Net<float> > net, std::vector<cv::Mat>* input_channels);
		void pyrDown(const std::vector<cv::Mat>& img_channels,float scale, std::vector<cv::Mat>* input_channels);
		void buildInputChannels(const std::vector<cv::Mat>& img_channels, const std::vector<BoundingBox>& boxes,
								const cv::Size& target_size, std::vector<cv::Mat>* input_channels);

		boost::shared_ptr< caffe::Net<float> > P_Net;
		boost::shared_ptr< caffe::Net<float> > R_Net;
		boost::shared_ptr< caffe::Net<float> > O_Net;
		//used by model 2 version
		boost::shared_ptr< caffe::Net<float> > L_Net;
		double                           img_mean;
		double                           img_var;
		cv::Size                         input_geometry_;
		int                              num_channels_;
		MODEL_VERSION                    model_version;
	};
}

#endif /* INCLUDE_MTCNN_FACEDETECTOR_HPP_ */
