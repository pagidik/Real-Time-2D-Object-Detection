#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/saturate.hpp>
#include "filter.h"
#include "csv_util.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <cmath>
using namespace cv;
using namespace std;



int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;
        // // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                        (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);
              
        // namedWindow("Video", 1);
        // namedWindow("Threshold",1);
 
        Mat frame;

        while(true) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
    
            vector<double> feature(0);
        
            if( frame.empty() ) {
                printf("frame is empty\n");
                // break;
            }   
            
                         
           
            Mat blur, hsv, sat, threshold;

            // // Blur the image
            frame.copyTo(blur);
            blur5x5(frame,blur);
            
            // //  convert to HSV 
            blur.copyTo( hsv );
            cvtColor(blur, hsv, COLOR_BGR2HSV);
            

            // // Saturate the image using custom function to move the  furthe away from the background
            saturate(hsv, sat, 40, 2, 0);
            sat.copyTo(threshold);   

            // // Threshold the image to create a binary image
            inRange(sat, Scalar(0, 20, 0), Scalar(179 , 255, 120), threshold);
            
            // Use Combination of shrinking and growing to clean up the image
            Mat shk, grw;
            // shrink to remove salt and pepper noise
            grassshrink(threshold, shk, 3 );
            // Grow 
            grassgrow(shk, grw, 10);
            // shrink
            grassshrink(grw, shk, 3 );
            // grow
            grassgrow(shk,grw, 5);
            // shrink
            grassshrink(grw, shk, 4);
            imshow("Before morph " , threshold);
            // close any remaining holes
            morphologyEx(shk, shk, MORPH_CLOSE, getStructuringElement(MORPH_RECT,Size(60,60)));
            imshow("after morph " , shk);


            // label the regions using connected components
            Mat labels, stats, centroids;
            int numlabels = connectedComponentsWithStats(shk, labels, stats, centroids , 8);
            
            
            RNG rng(12345);                
            vector<vector<Point>> contours;
            findContours( shk , contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            vector<RotatedRect> minRect( contours.size() );
            
            int instance = 0;
            int flag = 0 , flag2 = 0;
            int r = 0 , s = 0;

            for( size_t i = 0; i < contours.size(); i++ )
            {
                minRect[i] = minAreaRect( contours[i] );
                cv::Point2f vtx[4];
                minRect[i].points(vtx);

                if (contourArea(contours[i]) > 3000){
                                  
                    int count = 0;
                    for( int i = 0; i < 4; i++ ){
                        cv::Point2f pnt = vtx[i];
                       
                        // cout<< "X = " << pnt.x << ", Y = " << pnt.y << endl;
                        if(pnt.x > 10 && pnt.y > 10 && pnt.x< 1270 && pnt.y < 710 ){
                            count += 1;      
                            }
                        else{
                            continue;
                        }
                    }
 
                    if (count == 4 ){
                        instance = 1;
                        r = i;
                        
                        // check if there is a bigger contour availble for the image
                        if(contourArea(contours[i]) > 50000 && contourArea(contours[i]) < 700000){
                            flag2 = 1;
                            s = i;
                        }
                        
                    }
                }
            }

            
            double perarea = 0 ;  
            double hwratio = 0;

 
            Mat drawing = Mat::zeros( shk.size(), CV_8UC3 );
            for( size_t i = 0; i< contours.size(); i++ ){  
                Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );   
                // draw contours using the function below on a dark image 
                drawContours( drawing, contours, (int)i, color );                
            }  

            if (instance == 1){
                    
                if( flag2 == 1 ){
                    r = s;
                }
                Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
                
                
                // rotated rectangle
                Point2f rect_points[4];
                

                // extract four corner of the rectangle 
                minRect[r].points( rect_points );
                for ( int j = 0; j < 4; j++ )
                {
                    // draw each side of the rectangle and loop over 4 times to get a complete rectangle
                    line( frame, rect_points[j], rect_points[(j+1)%4], color );   
                }                       
            } 

            imshow("Contours", drawing);

            // calculate the percentage filled by the rectangle
                       
            perarea = (contourArea(contours[r]) / ( minRect[r].size.width * minRect[r].size.height ) );

            // calculate the height width ratio
            if ( minRect[r].size.width > minRect[r].size.height )
            {
                swap(minRect[r].size.width, minRect[r].size.height);
                // minRect[r].angle += 90.f;
                hwratio = ( minRect[r].size.width / minRect[r].size.height) ;

            }
            else {
                hwratio = ( minRect[r].size.height / minRect[r].size.width);
            }
            
            imshow("Video", frame);
            // imshow("shrink before threshold", shk);

            // Convert to CV_8U
            Mat img2;
            labels.convertTo(img2, CV_8UC3);

            // // Apply color map
            Mat3b out;
            applyCustomColormap(labels, out);
            // imshow("Labels" , out); 
            imshow("Clean output", shk);
            
            // // Calculate Hu Moments to make the images translation, scale , rotation and mirror invariants
       
            vector<Moments> mom(contours.size());
            
            mom[r] = moments(contours[r]);

            float u = mom[r].m02 / (mom[r].m00 * mom[r].m00);
            float v = mom[r].m20 / (mom[r].m00 * mom[r].m00);

            if ( v > u){
                v = mom[r].m20 / (mom[r].m00 * mom[r].m00);
                u = mom[r].m02 / (mom[r].m00 * mom[r].m00);

            }
            

            double alpha ; 
            // Point2f ctr = centroid[r];

            vector<Point2f> mc(contours.size());
            
            // for( int i = 0; i<contours.size(); i++)
            
            mc[r] = Point2f( mom[r].m10/mom[r].m00 , mom[r].m01/mom[r].m00 ); 

            // Calculate the orientation of the central axis using central moments
        
            alpha = 0.5 * atan2 ( 2 * mom[r].mu11 , mom[r].mu20 - mom[r].mu02  );
            
            Point2f p1  = Point2f(float(200 * cos(alpha) + mc[r].x ), float( 200 * sin(alpha) + mc[r].y)) ;
            Point2f p2  = Point2f(float(mc[r].x - 200 * cos(alpha)  ), float( mc[r].y - 200 * sin(alpha)  )) ;
            
            line( frame, p1, p2, Scalar(0, 0 , 255) );
            imshow("central Axis" , frame);

            // centralAngleStart[i] = Point2f ( mc[i].x + 200 * cos(central_angle[i]) , mc[i].y + 200 * sin(central_angle[i]) );
            // centralAngleEnd[i] = Point2f ( mc[i].x - 200 * cos(central_angle[i]) , mc[i].y - 200 * sin(central_angle[i]) );

            double huMoments[7];
            HuMoments(mom[r], huMoments);
            

            // // Log scale hu moments 
            for(int i = 0; i < 7; i++) {
            
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
            feature.push_back(huMoments[i]);
            }
            
            feature.push_back(hwratio);
            feature.push_back(perarea);
            feature.push_back(u);
            feature.push_back(v);
        
            // Read features from the feaeture vector file and store then in a vector of vectors.
            char file[] = "/home/kishore/PRCV/Project_3/feature_training.csv";
            vector<vector<float>> featureVector ;
            vector<vector<string>> labelVector ; 

            vector<char *> filenames;
            read_image_data_csv(file, filenames, featureVector , 0);

            // // Calculate standard Deviation
            vector<float> mean ;
            vector<float> stddev;
            map<float, string> difference;
            
            float sum[featureVector.at(0).size()] = {0} , ssd[featureVector.at(0).size()] = {0} , diff[featureVector.size()] = {0} ;

            for (int j = 0 ; j < featureVector.at(0).size(); j ++){
                for(int i = 0 ; i< featureVector.size(); i++){
                    sum[j] +=  featureVector.at(i).at(j);
                    
                }
            }
            for(int i = 0 ; i < featureVector.at(0).size(); i ++){
                mean.push_back(sum[i] / featureVector.size());
            }
            for (int j = 0 ; j < featureVector.at(0).size(); j ++){
                for (int i = 0 ; i< featureVector.size(); i++){
                    ssd[j] +=  (featureVector.at(i).at(j) - mean[j]) * (featureVector.at(i).at(j) - mean[j]);
                }
            }
            for (int i = 0 ; i < featureVector.at(0).size(); i ++){
                stddev.push_back(sqrt(ssd[i]/featureVector.size()));
            }
             
            for (int j = 0 ; j < featureVector.size(); j ++){
                for (int i = 0 ; i< featureVector.at(0).size(); i++){
                    diff[j] +=  abs( feature.at(i) - featureVector.at(j).at(i) ) / stddev.at(i);
                }
                difference.insert(pair<float, string>(diff[j], filenames[j]));
               
            } 
            
            float distance[1] ;
            string  f[1];
           

            auto it = difference.begin();
            for(int i = 0; i < 1 && it != difference.end(); ++i){
                ++it;
                if(it != difference.end()){
                    distance[i] = it->first;
                    f[i] = it->second ;
                }
                else{
                    std::cout << "not found";
                }
            }

//************************************ KNN *********************************************//
            vector<vector<vector<float>>> kdiff;
            
            int maxidx = 1;
            vector<float> idx;
            idx.push_back(maxidx);

            for(int i = 1; i< filenames.size()  ; i++){
 
                if(strcmp(filenames.at(i), filenames.at(i-1)) == 0){
                    idx.push_back( maxidx ); 
                    
                }
                else{
                    maxidx++;
                    idx.push_back( maxidx ); 
                }
            }


            for (int i = 1; i <= maxidx; i++){ // make the number of lists for the labels
                kdiff.push_back({{{0}}});
    

                for (int j=0; j< featureVector.size()-1; j++){ // Push the instances of each label into corresponding label index
                    if(idx.at(j) == i){
                        kdiff.at(i-1).push_back({{idx.at(j)}}); 
                    }                
                }
            }
            int rows[maxidx];
            rows[0] = 0;
            int counter = 0;
            // count number of rows and store it in an array to use as a reference for featureVector.
            for (int ix = 0; ix < maxidx; ix++){ 
                for(int jx = 1 ; jx < kdiff.at(ix).size(); jx++){
                    counter++;
                    
                }
                rows[ix+1] = counter;
            }
            // loop over all indexes store the distance metrics of a particular label in a map
            map<float, string> knn;
            char key = cv::waitKey(10);
            for (int ix = 0; ix < maxidx; ix++){ 
                map<float, string> kdmap;
                
                // loop over instances of each label
                for(int jx = 1 ; jx < kdiff.at(ix).size(); jx++){
                    float d[kdiff.at(ix).size()];
                    d[jx-1] = 0;

                    // distance metric
                    for (int k = 0; k< featureVector.at(0).size(); k++){
                       
                        d[jx-1]  +=  abs( feature.at(k) - featureVector.at(rows[ix] + jx).at(k) ) / stddev.at(k);
                        cout<< d[jx-1]<< endl;
                    }

                    // store the distances and the file name in a map
                    kdmap.insert(pair<float, string>(d[jx-1], filenames[rows[ix] + jx-1]));   
                
                }

                float kdistance[1] ;
                string  kf[1];
            

                auto kit = kdmap.begin();

                // for(auto itr = kdmap.begin();itr!=kdmap.end();itr++){
                //     // cout<< filenames[itr]<< endl;
                //     cout << itr->first << ": " << itr->second << endl;
                // }

                for(int i = 0; i < 1 && kit != kdmap.end(); ++i){
                    ++kit;
                    if(kit != kdmap.end()){
                        kdistance[i] = kit->first;
                        kf[i] = kit->second ;
                    }
                    else{
                        std::cout << "not found";
                    }
                }
                float kdist[maxidx];
                string kstring[maxidx];

                // sum the two lowest distances in a particular label and store it in an array
                kdist[ix] = kdistance[0] + kdistance[1];
                kstring[ix] = kf[0];

                knn.insert(pair<float, string>(kdist[ix], kstring[ix] ));  

            
            }
            for(auto itr = knn.begin();itr!=knn.end();itr++){
                // cout<< filenames[itr]<< endl;
                cout << itr->first << ": " << itr->second << endl;
                
            }

            float knndist[1] ;
            string  knnfile[1];
        
            auto knnit = knn.begin();
            for(int i = 0; i < 1 && knnit != knn.end(); ++i){
                ++knnit;
                if(knnit != knn.end()){
                    knndist[i] = knnit->first;
                    knnfile[i] = knnit->second ;
                }
                else{
                    std::cout << "not found";
                }
            }

            auto itr = knn.begin();

            putText(frame, itr->second, mc[r] , FONT_HERSHEY_TRIPLEX, frame.cols/500, Scalar({255,0,255}),frame.cols/300 );
            imshow("KNN detetction", frame);     



                
            // char key = cv::waitKey(10);
            if( key == 'q') {
                break;
            }  
            if ( key == 't') {
                flag = 1;
            } 
            
            
            // // Store features in a file
            if ( flag == 1){
                char label[15];
                cout << "Enter Label Name : ";
                cin >> label ;
                append_image_data_csv(file, label , feature, 0 ); 
                flag = 0;
            }  

        }
    
    return 0;
}
