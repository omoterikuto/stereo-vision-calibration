INCLUDE = -I/usr/local/include/opencv4/
LIBS    = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_calib3d -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudacodec -lopencv_xfeatures2d -lopencv_features2d -lopencv_cudafeatures2d


build: main.o main-cpu.o sad.o fast.o affine.o postprocess.o postprocess-cpu.o fast-cpu.o match.o match-cpu.o utils.o 
	nvcc -O3 -o main main.o main-cpu.o sad.o fast.o affine.o postprocess.o postprocess-cpu.o fast-cpu.o match.o match-cpu.o utils.o $(INCLUDE) $(LIBS)

main.o:main.cu
	nvcc -O3 -c main.cu $(INCLUDE) $(LIBS)


main-cpu.o:main-cpu.cu
	nvcc -O3  -c main-cpu.cu $(INCLUDE) $(LIBS)

sad.o:sad.cu
	nvcc -O3  -c sad.cu $(INCLUDE) $(LIBS)

fast.o:fast.cu
	nvcc -O3  -c fast.cu $(INCLUDE) $(LIBS)

affine.o:affine.cu
	nvcc -O3  -c affine.cu $(INCLUDE) $(LIBS) -use_fast_math

postprocess.o:postprocess.cu
	nvcc -O3  -c postprocess.cu $(INCLUDE) $(LIBS) -use_fast_math

postprocess-cpu.o:postprocess-cpu.cu
	nvcc -O3  -c postprocess-cpu.cu $(INCLUDE) $(LIBS)

fast-cpu.o:fast-cpu.cu
	nvcc -O3  -c fast-cpu.cu $(INCLUDE) $(LIBS)

match.o:match.cu
	nvcc -O3  -c match.cu $(INCLUDE) $(LIBS)

match-cpu.o:match-cpu.cu
	nvcc -O3  -c match-cpu.cu $(INCLUDE) $(LIBS)

utils.o:utils.cu
	nvcc -O3  -c utils.cu $(INCLUDE) $(LIBS)

clean:
	rm -f *.o