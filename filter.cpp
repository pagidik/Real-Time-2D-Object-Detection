/************************************************* Functions *************************************/
/*

	Author : Kishore Reddy Pagidi
	Compile using CMake file provided

    Run the following commands with the CMake file in the same directory
    
        1. cmake .
        2. make
        3. ./CV_PROJECT_3
    
    

*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/core/saturate.hpp>
#include <cmath>

#include "filter.h"

using namespace cv;
using namespace std;

/******************* Grayscale  *******************/


int grayscale(const cv::Mat &src, cv::Mat &dst){    
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){            
                cv::Vec3b value = src.at<cv::Vec3b>(i,j);
                dst.at<uchar>(i,j) = (value.val[0] + value.val[1]+value.val[2])/3;
        }
    }
    return 0;
}

/******************* Gaussian 5 x 5 Blur *******************/

int blur5x5(const cv::Mat &src, cv::Mat &dst){
    cv::Mat tmp;
    tmp.create(src.size(), src.type());
    
    for(int i = 0; i < src.rows; i++){
        for(int j = 2; j < src.cols-2; j++){
            for(int c = 0; c < 3; c++){
                tmp.at<cv::Vec3b>(i,j)[c] = src.at<cv::Vec3b>(i,j-2)[c]*0.1 + src.at<cv::Vec3b>(i,j-1)[c]*0.2 +src.at<cv::Vec3b>(i,j)[c]*0.4 + src.at<cv::Vec3b>(i,j+1)[c]*0.2 + src.at<cv::Vec3b>(i,j+2)[c]*0.1;                
            }           
        }    
    }

    for(int i = 2; i < src.rows-2; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<cv::Vec3b>(i,j)[c] = tmp.at<cv::Vec3b>(i-2,j)[c]*0.1 + tmp.at<cv::Vec3b>(i-1,j)[c]*0.2 +tmp.at<cv::Vec3b>(i,j)[c]*0.4 + tmp.at<cv::Vec3b>(i+1,j)[c]*0.2 + tmp.at<cv::Vec3b>(i+2,j)[c]*0.1;
            }            
        }   
    }
    return 0;
}

/******************* Sobel seperable filter X direction*******************/

int sobelX3x3(const cv::Mat &src, cv::Mat3s &dst){
    cv::Mat tmp;
    src.copyTo(tmp);

    for(int i = 0; i < src.rows; i++){
        for(int j = 1; j < src.cols-1; j++){
            for(int c = 0; c < 3; c++){
                tmp.at<cv::Vec3b>(i,j)[c] = src.at<cv::Vec3b>(i-1,j)[c]*0.25 + src.at<cv::Vec3b>(i,j)[c]*0.5 + src.at<cv::Vec3b>(i+1,j)[c]*0.25;
            }   
        }
    }

    for(int i = 1; i < src.rows-1; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<cv::Vec3s>(i,j)[c] = -tmp.at<cv::Vec3b>(i,j-1)[c]*1 +tmp.at<cv::Vec3b>(i,j)[c]*0 + tmp.at<cv::Vec3b>(i,j+1)[c]*1;
            }           
        }   
    }
    return 0;
}

/******************* Sobel seperable filter Y direction*******************/
int sobelY3x3(const cv::Mat &src, cv::Mat3s &dst){
    cv::Mat tmp;
    src.copyTo(tmp);

    for(int i = 0; i < src.rows; i++){
        for(int j = 1; j < src.cols-1; j++){
            for(int c = 0; c < 3; c++){
                tmp.at<cv::Vec3b>(i,j)[c] = src.at<cv::Vec3b>(i,j-1)[c]*0.25 +src.at<cv::Vec3b>(i,j)[c]*0.5 + src.at<cv::Vec3b>(i,j+1)[c]*0.25;                
            }            
        }    
    }
    for(int i = 1; i < src.rows-1; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<cv::Vec3s>(i,j)[c] = tmp.at<cv::Vec3b>(i-1,j)[c]*1 + tmp.at<cv::Vec3b>(i,j)[c]*0 - tmp.at<cv::Vec3b>(i+1,j)[c]*1;                
            }            
        }    
    }
    return 0;
}

/******************* Gradiant Magnitude ******************/
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){       
    for(int i = 0; i < sx.rows; i++){
        for(int j = 0; j < sx.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<cv::Vec3s>(i,j)[c] = sqrt(sx.at<cv::Vec3s>(i,j)[c]*sx.at<cv::Vec3s>(i,j)[c] + sy.at<cv::Vec3s>(i,j)[c]*sy.at<cv::Vec3s>(i,j)[c]);   
            }           
        }    
    }
    return 0;
}

/******************* Gradiant Orientation ******************/
int orientation( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){       
    for(int i = 0; i < sx.rows; i++){
        for(int j = 0; j < sx.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<cv::Vec3s>(i,j)[c] = atan2(sy.at<cv::Vec3s>(i,j)[c] , sx.at<cv::Vec3s>(i,j)[c]) * 57.295779513;   
            }           
        }    
    }
    return 0;
}

/******************* Blur Quantization ******************/

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){    
    int b = 255 / levels;
    unsigned char x, xt, xf;

    cv::Mat tmp;
    src.copyTo(tmp);

    blur5x5(src, tmp);

    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            for(int c = 0; c < 3; c++){
                x = tmp.at<cv::Vec3b>(i,j)[c];
                xt = x / b;
                xf = xt * b;
                dst.at<cv::Vec3b>(i,j)[c] = xf;

            }           
        }    
    }  
    return 0;

}

/******************* Cartoon Filter ******************/

int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold ){
    cv::Mat3s sx, sy, temp_mag;
    cv::Mat mag, bq, tmp;

    src.copyTo(sx);
    src.copyTo(sy);
    

    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    

    src.copyTo(temp_mag);
    magnitude(sx, sy, temp_mag);
    convertScaleAbs(temp_mag, mag);

    src.copyTo(bq);

    blurQuantize(src, bq, levels);

    bq.copyTo(tmp);

    for(int i = 0; i < bq.rows; i++){
        for(int j = 0; j < bq.cols; j++){
            for(int c = 0; c < 3; c++){                
                cv::Vec3b intensity = mag.at<cv::Vec3b>(i,j);
                int px = intensity.val[c];
                (px > magThreshold) ? dst.at<cv::Vec3b>(i,j)[c] = 0 : dst.at<cv::Vec3b>(i,j)[c] = tmp.at<cv::Vec3b>(i,j)[c];              
            }
        }    
    }
    return 0;
}

/******************* Negative ******************/

int negative(const cv::Mat &src, cv::Mat &dst){

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){                
                dst.at<cv::Vec3b>(i,j)[c] = 255 - src.at<cv::Vec3b>(i,j)[c];                              
            }
        }    
    }
    return 0;
}

/******************************* Addjust Brightness ****************************/
int brightness(cv::Mat &src, cv::Mat &dst, int b) {
    
    src.convertTo(dst, -1, 1, b*10);

    return 0;
   
}

/**************************** Laws Filter L5 E5 *******************************/


int lawsl5e5(const cv::Mat &src, cv::Mat3s &dst){
    cv::Mat tmp;
    tmp.create(src.size(), src.type());
    
    for(int i = 0; i < src.rows; i++){
        for(int j = 2; j < src.cols-2; j++){
            for(int c = 0; c < 3; c++){
                tmp.at<cv::Vec3b>(i,j)[c] = src.at<cv::Vec3b>(i,j-2)[c]*0.0625 + src.at<cv::Vec3b>(i,j-1)[c]*0.25 +src.at<cv::Vec3b>(i,j)[c]*0.375 + src.at<cv::Vec3b>(i,j+1)[c]*0.25 + src.at<cv::Vec3b>(i,j+2)[c]*0.0625;                
            }           
        }    
    }

    for(int i = 2; i < src.rows-2; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<cv::Vec3s>(i,j)[c] = tmp.at<cv::Vec3b>(i-2,j)[c]*0.3333 + tmp.at<cv::Vec3b>(i-1,j)[c]*0.6666 +tmp.at<cv::Vec3b>(i,j)[c]*0 - tmp.at<cv::Vec3b>(i+1,j)[c]*0.6666 - tmp.at<cv::Vec3b>(i+2,j)[c]*0.3333;
            }            
        }   
    }
    return 0;
}

/******************************** 2d Histogram *********************************/

int histogram2d(cv::Mat &src, float histogram[], int Hsize){

    int rix = 0 ,gix = 0, sum = 0;

    for(int i= 0 ; i < src.rows; i++){
        for(int j = 0 ; j< src.cols; j++){
            
            sum = src.at<cv::Vec3b>(i,j)[0] + src.at<cv::Vec3b>(i,j)[1] + src.at<cv::Vec3b>(i,j)[2] +1;
            rix = ( Hsize * src.at<cv::Vec3b>(i,j)[2] )/sum;
            gix = ( Hsize * src.at<cv::Vec3b>(i,j)[1] )/sum;
            histogram[ rix * Hsize + gix ]++;

        }
    }
    return 0;
}

/********************************* 3d Histogram ***********************************/

int histogram3d(cv::Mat &src, float histogram[], int Hsize){

    int rix = 0 , gix = 0, bix = 0, divisor = 256/Hsize;

    for(int i= 0 ; i < src.rows; i++){
        for(int j = 0 ; j< src.cols; j++){
           
            rix = (  src.at<cv::Vec3b>(i,j)[2] )/divisor;
            gix = ( src.at<cv::Vec3b>(i,j)[1] )/divisor;
            bix = (  src.at<cv::Vec3b>(i,j)[0] )/divisor;
            histogram[ rix * Hsize * Hsize + gix * Hsize + bix ]++;
                
        }
    }

    
    return 0;
}


/******************* Sobel seperable filter on 1D image in X direction*******************/


int sobelX3x3_1d(const cv::Mat &src, cv::Mat &dst){
    cv::Mat tmp;
    src.copyTo(tmp);

    for(int i = 0; i < src.rows; i++){
        for(int j = 1; j < src.cols-1; j++){
            // for(int c = 0; c < 3; c++){
                tmp.at<uchar>(i,j)  = src.at<uchar>(i-1,j) *0.25 + src.at<uchar>(i,j) *0.5 + src.at<uchar>(i+1,j) *0.25;
            // }   
        }
    }

    for(int i = 1; i < src.rows-1; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<uchar>(i,j)  = -tmp.at<uchar>(i,j-1) *1 +tmp.at<uchar>(i,j) *0 + tmp.at<uchar>(i,j+1) *1;
            }           
        }   
    }
    return 0;
}

/******************* Sobel seperable filter on 1D image in Y direction*******************/

int sobelY3x3_1d(const cv::Mat &src, cv::Mat &dst){
    cv::Mat tmp;
    src.copyTo(tmp);

    for(int i = 0; i < src.rows; i++){
        for(int j = 1; j < src.cols-1; j++){
    
                tmp.at<uchar>(i,j) = src.at<uchar>(i,j-1) *0.25 +src.at<uchar>(i,j) *0.5 + src.at<uchar>(i,j+1) *0.25;                
                        
        }    
    }
    for(int i = 1; i < src.rows-1; i++){
        for(int j = 0; j < src.cols; j++){
            for(int c = 0; c < 3; c++){
                dst.at<uchar>(i,j)  = tmp.at<uchar>(i-1,j) *1 + tmp.at<uchar>(i,j) *0 - tmp.at<uchar>(i+1,j) *1;                
            }            
        }    
    }
    return 0;
}


/************************ Gradient Magnitude on 1D Image ****************************/


int magnitude_1d( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){       
    for(int i = 0; i < sx.rows; i++){
        for(int j = 0; j < sx.cols; j++){
       
                dst.at<uchar>(i,j)  = sqrt(sx.at<uchar>(i,j) *sx.at<uchar>(i,j)  + sy.at<uchar>(i,j) *sy.at<uchar>(i,j) );   
                  
        }    
    }
    return 0;
}


/**************************** Saturation  *****************************/

int saturate(const cv::Mat &src, cv::Mat &dst, int s, float alpha, float beta ){

    src.copyTo(dst);

    for( int i = 0; i < src.rows; i++ ) {
        for( int j = 0;  j < src.cols; j ++ ) {
        
            if( src.at<Vec3b>(i,j)[1] > s){
                dst.at<Vec3b>(i,j)[1] =
                saturate_cast<uchar>(alpha*src.at<Vec3b>(i,j)[1] + beta) ;
                // cout << i << " , " << j  << " ," ;
            }
        }
    }
    return 0;
}


/*************************** Grassfire Transform ***********************/
int grassshrink(const cv::Mat &src , cv::Mat &dst , int n){

    cv::Mat tmp;
    src.copyTo(tmp);
    cv::Mat dist;


    //Pass 1
    for( int i = 1; i < src.rows; i++ ) {
        for( int j = 1;  j < src.cols; j ++ ) {
        
            if( src.at<uchar>(i,j) == 0  ){
                tmp.at<uchar>(i,j) = 0;
            }
            else {
                tmp.at<uchar>(i,j) = saturate_cast<uchar>(1 + std::min(tmp.at<uchar>(i-1,j) , tmp.at<uchar>(i,j-1)));
                
            }
  
        }
    }

    //Pass 2   
    tmp.copyTo(dist);

    for( int i = src.rows-1; i >0; --i ) {
        for( int j = src.cols-1;  j >0 ; --j ) {
            
        
            if( tmp.at<uchar>(i,j) == 0 ){
                dist.at<uchar>(i,j) = 0;
            }
            else {

                dist.at<uchar>(i,j) = std::min(dist.at<uchar>(i,j), saturate_cast<uchar>( 1 + std::min(dist.at<uchar>(i+1,j) , dist.at<uchar>(i,j+1))) );
            }            
    
        }
    }


    // shrink
    src.copyTo(dst);

    for( int i = 0; i < src.rows-1; i++ ) {
        for( int j = 0;  j < src.cols-1; j++ ) {
        
            if( dist.at<uchar>(i,j) < n + 1  ){
                dst.at<uchar>(i,j) = 0; 
            }

            else {
                continue;
            }
                
    
        }
    }


    return 0; 
}



int grassgrow(const cv::Mat &src , cv::Mat &dst, int n){

    cv::Mat tmp;
    src.copyTo(tmp);

    cv::Mat dist;

    //Pass 1
    for( int i = 1; i < src.rows; i++ ) {
        for( int j = 1;  j < src.cols; j ++ ) {
        
            if( src.at<uchar>(i,j) == 255  ){
                tmp.at<uchar>(i,j) = 0;
            }
            else {
                tmp.at<uchar>(i,j) = saturate_cast<uchar>(1 + std::min(tmp.at<uchar>(i-1,j) , tmp.at<uchar>(i,j-1)));
                
            }
  
        }
    }

    // imshow("Grow Pass 1" , tmp);
    //Pass 2   
    tmp.copyTo(dist);

    for( int i = src.rows-1; i >0; --i ) {
        for( int j = src.cols-1;  j >0 ; --j ) {
            
        
            if( tmp.at<uchar>(i,j) == 0 ){
                dist.at<uchar>(i,j) = 0;
            }
            else {

                dist.at<uchar>(i,j) = std::min(dist.at<uchar>(i,j), saturate_cast<uchar>( 1 + std::min(dist.at<uchar>(i+1,j) , dist.at<uchar>(i,j+1))) );
            }            
    
        }
    }
    // Grow
    src.copyTo(dst);

    for( int i = 0; i < src.rows-1; i++ ) {
        for( int j = 0;  j < src.cols-1; j ++ ) {
        
            if( dist.at<uchar>(i,j) < n + 1  ){
                dst.at<uchar>(i,j) = 255;
             
            }
            else {
                continue;
            }
                
    
        }
    }

    return 0; 
}

void applyCustomColormap(const cv::Mat1i& src, cv::Mat3b& dst)
{
    // Create JET colormap

    double m;
    minMaxLoc(src, nullptr, &m);
    m++;

    int n = ceil(m / 4);
    Mat1d u(n*3-1, 1, double(1.0));

    for (int i = 1; i <= n; ++i) { 
        u(i-1) = double(i) / n; 
        u((n*3-1) - i) = double(i) / n;
    }

    vector<double> g(n * 3 - 1, 1);
    vector<double> r(n * 3 - 1, 1);
    vector<double> b(n * 3 - 1, 1);
    for (int i = 0; i < g.size(); ++i)
    {
        g[i] = ceil(double(n) / 2) - (int(m)%4 == 1 ? 1 : 0) + i + 1;
        r[i] = g[i] + n;
        b[i] = g[i] - n;
    }

    g.erase(remove_if(g.begin(), g.end(), [m](double v){ return v > m;}), g.end());
    r.erase(remove_if(r.begin(), r.end(), [m](double v){ return v > m; }), r.end());
    b.erase(remove_if(b.begin(), b.end(), [](double v){ return v < 1.0; }), b.end());

    Mat1d cmap(m, 3, double(0.0));
    for (int i = 0; i < r.size(); ++i) { cmap(int(r[i])-1, 2) = u(i); }
    for (int i = 0; i < g.size(); ++i) { cmap(int(g[i])-1, 1) = u(i); }
    for (int i = 0; i < b.size(); ++i) { cmap(int(b[i])-1, 0) = u(u.rows - b.size() + i); }

    Mat3d cmap3 = cmap.reshape(3);

    Mat3b colormap;
    cmap3.convertTo(colormap, CV_8U, 255.0);


    // Apply color mapping
    dst = Mat3b(src.rows, src.cols, Vec3b(0,0,0));
    for (int r = 0; r < src.rows; ++r)
    {
        for (int c = 0; c < src.cols; ++c)
        {
            dst(r, c) = colormap(src(r,c));
        }
    }
}

int regionthreshold(const cv::Mat &src , cv::Mat &dst, int numlabels , int thres ){

    

    float *hist1d = new float[numlabels];

    for(int i = 0; i < numlabels ; i++ ){
        hist1d[i] = 0;
        
    }
    for( int i = 0; i < src.rows ; i ++){
        for ( int j = 0 ; j < src.cols ; j++){
            int id = src.at<int>(i,j) ;
            hist1d[ id ]++; 
        }
    }

    for (int i = 0; i < numlabels; i++){
        // cout<< hist1d[i] << " ; ";
    }

    for (int id = 0; id < numlabels; id++){
        
        if(hist1d[id] < thres  ){
            for( int i = 0; i < src.rows ; i ++){
                for ( int j = 0 ; j < src.cols ; j++){
                    if( src.at<int>(i,j) == id ){
                        dst.at<uchar>(i,j) = 0;

                    }
                    
                }
            } 
            
        }
    }

    return 0 ;

}


