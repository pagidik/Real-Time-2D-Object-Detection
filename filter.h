
/************************************************* Function Header File *************************************/
/*

	Author : Kishore Reddy Pagidi
	Compile using CMake file provided

    Run the following commands with the CMake file in the same directory
    
        1. cmake .
        2. make
        3. ./CV_PROJECT_3
    
    

*/

int grayscale(const cv::Mat &src, cv::Mat &dst);

int blur5x5(const cv::Mat &src, cv::Mat &dst);

int sobelX3x3(const cv::Mat &src, cv::Mat3s &dst);

int sobelY3x3(const cv::Mat &src, cv::Mat3s &dst);

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

int negative(const cv::Mat &src, cv::Mat &dst);

int brightness(cv::Mat &src, cv::Mat &dst, int b);

int orientation( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int lawsl5e5(const cv::Mat &src, cv::Mat3s &dst);

int histogram2d(cv::Mat &src, float histogram[], int Hsize);

int histogram3d(cv::Mat &src, float histogram[], int Hsize);

int readDir(char argv[1] , std::string path[] );

int sobelX3x3_1d(const cv::Mat &src, cv::Mat &dst);

int sobelY3x3_1d(const cv::Mat &src, cv::Mat &dst);

int magnitude_1d( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int saturate(const cv::Mat &src, cv::Mat &dst, int s, float alpha, float beta );

int grassshrink(const cv::Mat &src , cv::Mat &dst, int n);

int grassgrow(const cv::Mat &src , cv::Mat &dst, int n);

void applyCustomColormap(const cv::Mat1i& src, cv::Mat3b& dst);

int regionthreshold(const cv::Mat &src , cv::Mat &dst, int numlabels , int thres );