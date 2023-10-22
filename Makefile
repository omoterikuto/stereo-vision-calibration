INCLUDE = -I/usr/local/include/opencv4/
LIBS    = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_calib3d -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudacodec -lopencv_xfeatures2d -lopencv_features2d -lopencv_cudafeatures2d


build-calibration:
	make --directory calibration build

clean-calibration:
	make --directory calibration clean

build-stereo-vision:
	nvcc -o stereo-vision/app stereo-vision/*.c* -lpng

build: 
	make build-stereo-vision
	make build-calibration

run-calibration:
	calibration/main 000000_10_gray_L.bmp 000000_10_gray_R.bmp

run-stereo-vision:
	stereo-vision/app calibration-result/000000_10_gray_L.png calibration-result/000000_10_gray_R.png stereo-vision-result/output.png

run:
	make run-calibration
	make run-stereo-vision