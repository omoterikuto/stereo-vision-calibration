#define IMG_SIZEX 2736
#define IMG_SIZEY 1824
#define SAD_dc 10
#define X_STEP 20
#define Y_STEP 200
#define X_MARGIN 20
#define BLOCK_SHARED_X 128
#define MAX_RESULT 1000
#define SEARCH_AREA_X 10
#define F 400
#define NUM_SAD ((IMG_SIZEX-2*X_MARGIN)/X_STEP)
#define NUM_LINES 5

#define MAX_KEYPOINT 50000

struct sad_match_t {
    int y1;
    int x1;
    int y2;
    int x2;
    int distance;
};

struct least_squares_t{
    float a;
    float b;
};


struct keypoint_t {
    int y;
    int x;
    unsigned long long int feature1;
    unsigned long long int feature2;

};

struct brief_t {
    int y;
    int x;
    int feature[256];
};

struct match_t {
    int y1;
    int x1;
    int y2;
    int x2;
    int distance;
    unsigned long long int feature1_1;
    unsigned long long int feature1_2;
    unsigned long long int feature2_1;
    unsigned long long int feature2_2;
};
