#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <iostream>

#include<dirent.h>
#include<unistd.h>

#include <MTCNN/FaceDetector.hpp>

void readFileList(const char* basePath, std::vector<std::string>& imgFiles)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    
    if( (dir=opendir(basePath)) == NULL)
    {
        return ;
    }
    
    while( (ptr = readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".") == 0 ||
            strcmp(ptr->d_name, "..") == 0)
            continue;
        else if(ptr->d_type == 8)//file 
        {
            int len = strlen(ptr->d_name);
            if(ptr->d_name[len-1] == 'g' && ptr->d_name[len-2] == 'p' && ptr->d_name[len-3] == 'j')
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if(ptr->d_type == 10)/// link file
        {
            int len = strlen(ptr->d_name);
            if(ptr->d_name[len-1] == 'g' && ptr->d_name[len-2] == 'p' && ptr->d_name[len-3] == 'j')
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if(ptr->d_type == 4)//dir
        {
            memset(base, '\0', sizeof(base));
            strcpy(base, basePath);
            strcat(base, "/");
            strcat(base, ptr->d_name);
            readFileList(base, imgFiles);
        }
    }
    closedir(dir);
}



int main(int argc, char **argv) {
    //std::cout << "Hello, world!" << std::endl;
    mtcnn::FaceDetector fd("models/v2", mtcnn::FaceDetector::MODEL_V1);
    
    std::vector<std::string> imgList;
    readFileList("/media/rcnn/DATA/data/CASIA-WebFace", imgList);
    for(int l = 0; l < imgList.size(); l ++)
    {
        cv::Mat testImg = cv::imread(imgList[l]);
        
        std::vector<mtcnn::FaceDetector::BoundingBox> res = fd.Detect(testImg, mtcnn::FaceDetector::BGR, mtcnn::FaceDetector::ORIENT_UP ,20, 0.6, 0.7, 0.7);
        std::cout<< "detected face NUM : " << res.size() << std::endl;
        for(int k = 0; k < res.size(); k++)
        {
            cv::rectangle(testImg, cv::Point(res[k].x1, res[k].y1), cv::Point(res[k].x2, res[k].y2), cv::Scalar(0, 255, 255), 2);
            for(int i = 0; i < 5; i ++)
                cv::circle(testImg, cv::Point(res[k].points_x[i], res[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow("test", testImg);
        cv::waitKey();
    }
    return 0;
}
