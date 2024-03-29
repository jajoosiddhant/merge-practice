int main(int argc, char** argv)
{
	//Testing git fetch command
	
	Mat src;
	Mat src_half, src_res, gray;
	Mat contrast;	
	Mat hls, hsv, mask;
	Mat white, yellow;
	Mat hello;
	Mat blur, edge;
	vector<Vec4i> lines;
	
	while(capture.read(src))
	{	

		//Preprocess frames
		//Reduce the resolution by 2
		pyrDown(src, src_res);
		
		//Crop the image to get bottom half because sky is not required. Thus eliminating half the pixels.
		src_half = src_res( Rect( 0, src_res.rows/2, src_res.cols, src_res.rows/2) );
		imshow("Original", src_half);		
		
		//Convert to grayscale
		cvtColor(src_half, gray, COLOR_BGR2GRAY);
		
		//Increase Contrast to detect white lines and remove any disturbance due to road color.
		equalizeHist(gray, contrast);
		//imshow("Contrast Image", contrast);
		//imshow("Half image and resolution and gray", gray);
		
		
		//Convert Original Image to HLS
		//Shows white as yellow.
		cvtColor(src_half, hls, COLOR_BGR2HLS);
		//imshow("HLS", hls);
			
		
		//Convert Original Image to HSV
		//Shows yellow as yellow.
		cvtColor(src_half, hsv, COLOR_BGR2HSV);
		//imshow("HSV", hsv);
		
		inRange(hls, Scalar(20,100,0), Scalar(40,255,50), white);
		//imshow("HLS white", white);
		
		inRange(hsv, Scalar(20,90,100), Scalar(40,255,150), yellow);
		//imshow("HSV YELLOW", yellow);

		bitwise_or(white, yellow, mask);
		//imshow("Mask", mask);
		
		bitwise_and(contrast, mask, hello);
		imshow("Hello", hello);
		
		//Applying gaussian filter to reduce noise followed by canny transform for edge detection.
		GaussianBlur( hello, blur, Size(5,5), 0, 0, BORDER_DEFAULT );
		Canny(blur, edge, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2, 3, true);	//Can change to false
		

		imshow("Canny", edge);


		//Detect and Draw Lines
		
		HoughLinesP(edge, lines, 1, CV_PI/180, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP );
		for( size_t i = 0; i < lines.size(); i++ )
		{
			Vec4i l = lines[i];
			line( src_half, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		}

		imshow("Final", src_half);


//		frame = src.clone();
//		cvtColor(edge, test, CV_GRAY2RGB);
//		cvtColor(edge, test1, CV_GRAY2RGB);
		
		//Detecting lanes and drawing lines.
		//test = detect_draw_lanes(gray, src);

		//Detecting Vehicles.
//		test = detect_draw_vehicle(gray, gray);

		//imshow("source", src);
		//imshow("Detected lines", test);
//		imshow("Detected Vehicles", test);
//		sprintf(out, "./frames_snapshot/frame_%d.jpg", k);
//        imwrite(out, frame);
//        outputvideo.write(frame);

        
        k++;
        cnt_frame++;
        
        c = cvWaitKey(10);
        if( c == 27 ) break;
	
	}
	
	gettimeofday(&stop, NULL);    
    cout << "Duration: " << (stop.tv_sec - start.tv_sec) << " seconds." << endl;
	cout << "Average Frame rate = " << cnt_frame/((int)(stop.tv_sec - start.tv_sec)) << endl;
	return 0;
}


Mat preprocess(Mat src)
{
	Mat src_half, src_res, gray;
	
	//Crop the image to get bottom half because sky is not required. Thus eliminating half the pixels.
	src_half = src( Rect( 0, src.rows/2, src.cols, src.rows/2) );
	
	//Reduce the resolution by 2
	pyrDown(src_half, src_res);
		
	return src_res;
}




/**
 * @brief This function detects straight lines and draws lines over those detected lines.
 * @param edge The edge detected frame.
 * @param frame The frame on which the lines are drawn.
 * @return Mat frame. 
 */
Mat detect_draw_lanes(Mat src_res, Mat frame)
{
	Mat gray, equalize, hls, hls_filter, blur, edge;
	Mat white, yellow, hello, mask;
	
	//Convert the image to grayscale
	cvtColor(src_res, gray, CV_RGB2GRAY);
	imshow("Gray Image", gray);
	
	//Increase Contrast
	equalizeHist(gray, equalize);
	imshow("Contrast Image", equalize);
	
	//Convert Original Image to HSL
	cvtColor(src_res, hls, COLOR_BGR2HLS);
	imshow("HLS", hls);

//	inRange(hls, Scalar(0,50,0), Scalar(0,255,0), white);
//	imshow("HLS white", white);
	
//	inRange(hls, Scalar(0,63,48), Scalar(62,74,57), white);
//	imshow("HLS Yellow", yellow);
	
//	bitwise_or(white, yellow, mask)
//	imshow("Mask", mask);
	
//	bitwise_and(equalize, white, hello);
//	imshow("Hello", hello);
	//inRange(hls, Scalar());
	
	
	
	
	
	//Applying gaussian filter to reduce noise followed by canny transform for edge detection.
	GaussianBlur( gray, blur, Size(5,5), 0, 0, BORDER_DEFAULT );
	Canny(blur, edge, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2, 3, true);	//Can change to false
			
	//Another method is to split and just obtain the Green Channel.
	//split(src, bgr);
	//GaussianBlur( bgr[1], blur, Size(5,5), 0, 0, BORDER_DEFAULT )
	//Canny(blur, edge, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2, 3, true);
	
	
	//Crop the image over here
	
	
	//Detect and Draw Lines
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI/180, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP );
	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		line( frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
	}
	
	return frame;
}


/**
 * @brief This function detects vehicles and draws rectangles over those detected vehicles.
 * @param edge The edge detected frame.
 * @param frame The frame on which the lines are drawn.
 * @return Mat frame. 
 */
Mat detect_draw_vehicle(Mat edge, Mat frame)
{
	vector<Rect> vehicle;
	
	vehicle_cascade.detectMultiScale(edge, vehicle);
	
	for ( size_t i = 0; i < vehicle.size(); i++ )
    {
		//vehicle[i].y += 359;
		//vehicle[i].x += 359;
		rectangle(frame, vehicle[i], CV_RGB(255, 0, 0) );		
    }

	return frame;
}



