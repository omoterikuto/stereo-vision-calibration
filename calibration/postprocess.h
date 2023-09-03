least_squares_t least_squares(sad_match_t *sad_match, int num_sad);
float estimateRotationY(least_squares_t *lines, int num_lines);
float estimateRotationX(least_squares_t *lines, int num_lines);
float estimateRotationZ(least_squares_t *lines, int num_lines);
float estimateTranslationY(least_squares_t *lines, int num_lines);
float estimateTranslationZ(least_squares_t *lines, int num_lines);

__global__ void kernel_transform_ry(unsigned char* SrcImg, unsigned char* DstImg, float beta);
__global__ void kernel_transform_rx(unsigned char* SrcImg, unsigned char* DstImg, float alpha);
__global__ void kernel_transform_rz(unsigned char* SrcImg, unsigned char* DstImg, float gamma);
__global__ void kernel_transform_ty(unsigned char* SrcImg, unsigned char* DstImg, float dy);
__global__ void kernel_transform_tz(unsigned char* SrcImg, unsigned char* DstImg, float dz);

