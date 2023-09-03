cv::Mat read_images(std::string path, cv::cuda::Stream *stream);
cv::Mat read_images_cpu(std::string path);
bool cmp_match(cv::DMatch &p, cv::DMatch &q); //マッチングペアsort用関数
bool cmp_match2(const struct match_t& p, const struct match_t& q);//マッチングペアsort用関数
void draw_SAD_CUDA(int C, cv::Mat img_L, cv::Mat img_R, sad_match_t *sad_match, int num_matched);
