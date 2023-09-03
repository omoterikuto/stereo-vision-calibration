cv::Mat read_images(std::string path);
int calc_FAST_cuda(cv::Mat img_L, cv::Mat img_R, cv::Mat* out_L, cv::Mat* out_R);
bool cmp_match2(const struct match_t& p, const struct match_t& q);