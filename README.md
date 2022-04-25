# Real Time 2D Object Detection 

The scope of this project is to design a filter from scratch to run on real-time videos. In this project, I have implemented the following: 

1. Converted live video into grayscale.
2. Blurred the image using Gaussian 5x5 blur.
3. Applied Sobel filter in a separable form to reduce computational costs
4. Computed the gradient magnitude from the Sobel filter.
5. Reduced the number of colors in the image using blur quantification.
6. Produced a cartoon effect on the live video by darkening the edges using a certain threshold value.
7. Used additional filters to convert the video into negative.
8. Defined a function to increase and decrease the brightness with a keypress.
9. Let the user save short video sequences with the filters.
10. A negative image effect will be visible on a real-time video. 

Wiki Khoury link : https://wiki.khoury.northeastern.edu/display/~kishore005/Real+Time+2D+Object+Detection

Compile using cmake file.
1. cmake .
2. make
3. ./CV_PROJECT_3
