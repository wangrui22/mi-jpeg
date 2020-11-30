#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>


// int check_y() {
//     std::string quat_gpujpeg_f = "/home/wangrui22/projects/mi-jpeg/data/gpujpeg-data-16384-0.raw";
//     std::string quat_gjpegencoder_f = "/home/wangrui22/projects/mi-jpeg/data/cjpegencoder-data-16384-0.raw";

//     const int mcu_count = 16384;
//     unsigned int len = mcu_count*64;
//     unsigned char* quat_gpujpeg = new unsigned char[len];
//     unsigned char* quat_gjpegencoder = new unsigned char[len];

//     {
//         unsigned char* quat_gpujpeg_raw = new unsigned char[len];
//         std::ifstream in(quat_gpujpeg_f, std::ios::binary | std::ios::in);
//         if (!in.is_open()) {
//             return -1;
//         }
//         in.read((char*)quat_gpujpeg_raw, len);
//         in.close();

//         int width = 1024;
//         int height = 1024;
//         int mcu_w = width/8;
//         int mcu_h = height/8;
//         int sidx = 0;
//         for (int i=0; i<mcu_count; ++i) {
//             int y0 = i/mcu_w;
//             int x0 = i-y0*mcu_w;
//             y0*=8;
//             x0*=8;
//             int y1 = y0+8;
//             int x1 = x0+8;
//             y1 = y1 < height ? y1 : height;
//             x1 = x1 < width ? x1 : width;
//             int idx = 0;
//             float r,g,b,y,u,v;
//             for (int iy=y0; iy<y1; ++iy) {
//                 for (int ix=x0; ix<x1; ++ix) {
//                     idx = iy*width + ix;
//                     quat_gpujpeg[sidx] = quat_gpujpeg_raw[idx];
//                     ++sidx;
//                 }
//             }
//         }
//         assert(sidx == mcu_count*64);

//     }

//     {
//         std::ifstream in(quat_gjpegencoder_f, std::ios::binary | std::ios::in);
//         if (!in.is_open()) {
//             return -1;
//         }
//         in.read((char*)quat_gjpegencoder, len);
//         in.close();
//     }

//     unsigned num = 0;

//     for (int m=0; m<mcu_count; ++m) {
//         // std::cout << "\n<><><><><><><><>mcu: " << m << "<><><><><><><><>\ngpujeg:\n";
//         // int idx = m*64;
//         // for (int y=0; y<8; ++y) {
//         //     for (int x=0; x<8; ++x) {
//         //         std::cout << (int)quat_gpujpeg[idx++] << " ";
//         //     }
//         //     std::cout << "\n";
//         // }
//         // std::cout << "cjpegencoder\n";
//         // idx = m*64;
//         // for (int y=0; y<8; ++y) {
//         //     for (int x=0; x<8; ++x) {
//         //         std::cout << (int)quat_gjpegencoder[idx++] << " ";
//         //     }
//         //     std::cout << "\n";
//         // }

//         for (int i=0; i<64; ++i) {
//             int idx = m*64+i;
//             unsigned char q0 = quat_gpujpeg[idx];
//             unsigned char q1 = quat_gjpegencoder[idx];
//             if (q0 != q1) {
//                 std::cerr << "m: " << m << ", id: " << i << ", val: "<< (int)q0 << "/" << (int)q1 << std::endl;
//                 ++num;
//             }

//         }
//     }

//     std::cout << "diff sum: " << num << "\n";

//     return 0;
// }

int check_yuv_and_quat() {
    std::string file_gpujpeg_quat = "/home/wangrui22/projects/mi-jpeg/data/gpujpeg-quat-16384-0.raw";
    std::string file_gjpegencoder_quat = "/home/wangrui22/projects/mi-jpeg/data/cjpegencoder-quat-16384-0.raw";

    std::string file_gpujpeg_y_data = "/home/wangrui22/projects/mi-jpeg/data/gpujpeg-data-16384-0.raw";
    std::string file_gjpegencoder_y_data = "/home/wangrui22/projects/mi-jpeg/data/cjpegencoder-data-16384-0.raw";

    const int mcu_count = 16384;
    unsigned int len = mcu_count*64;
    short* quat_gpujpeg = new short[len];
    short* quat_gjpegencoder = new short[len];
    unsigned char* data_y_gpujpeg = new unsigned char[len];
    unsigned char* data_y_gjpegencoder = new unsigned char[len];

    {
        std::ifstream in(file_gpujpeg_quat, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            return -1;
        }
        in.read((char*)quat_gpujpeg, len*2);
        in.close();
    }

    {
        std::ifstream in(file_gjpegencoder_quat, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            return -1;
        }
        in.read((char*)quat_gjpegencoder, len*2);
        in.close();
    }

    {
        unsigned char* quat_gpujpeg_raw = new unsigned char[len];
        std::ifstream in(file_gpujpeg_y_data, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            return -1;
        }
        in.read((char*)quat_gpujpeg_raw, len);
        in.close();

        int width = 1024;
        int height = 1024;
        int mcu_w = width/8;
        int mcu_h = height/8;
        int sidx = 0;
        for (int i=0; i<mcu_count; ++i) {
            int y0 = i/mcu_w;
            int x0 = i-y0*mcu_w;
            y0*=8;
            x0*=8;
            int y1 = y0+8;
            int x1 = x0+8;
            y1 = y1 < height ? y1 : height;
            x1 = x1 < width ? x1 : width;
            int idx = 0;
            float r,g,b,y,u,v;
            for (int iy=y0; iy<y1; ++iy) {
                for (int ix=x0; ix<x1; ++ix) {
                    idx = iy*width + ix;
                    data_y_gpujpeg[sidx] = quat_gpujpeg_raw[idx];
                    ++sidx;
                }
            }
        }
        assert(sidx == mcu_count*64);

    }

    {
        std::ifstream in(file_gjpegencoder_y_data, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            return -1;
        }
        in.read((char*)data_y_gjpegencoder, len);
        in.close();
    }

    unsigned num = 0;

    for (int m=0; m<mcu_count; ++m) {
        std::cout << "\n<><><><><><><><>mcu: " << m << "<><><><><><><><>\ngpujeg data y:\n";

        {
            int idx = m*64;
            for (int y=0; y<8; ++y) {
                for (int x=0; x<8; ++x) {
                    std::cout << (int)data_y_gpujpeg[idx++] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "cjpegencoder data y: \n";
            idx = m*64;
            for (int y=0; y<8; ++y) {
                for (int x=0; x<8; ++x) {
                    std::cout << (int)data_y_gjpegencoder[idx++] << " ";
                }
                std::cout << "\n";
            }

            for (int i=0; i<64; ++i) {
                int idx = m*64+i;
                unsigned char q0 = data_y_gpujpeg[idx];
                unsigned char q1 = data_y_gjpegencoder[idx];
                if (q0 != q1) {
                    std::cerr << "m: " << m << ", id: " << i << ", val: "<< (int)q0 << "/" << (int)q1 << std::endl;
                    ++num;
                }
            }
        }
        

        std::cout << "\n<><><><><><><><>mcu: " << m << "<><><><><><><><>\ngpujeg quat:\n";
        {
            int idx = m*64;
            for (int y=0; y<8; ++y) {
                for (int x=0; x<8; ++x) {
                    std::cout << quat_gpujpeg[idx++] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "cjpegencoder quat: \n";
            idx = m*64;
            for (int y=0; y<8; ++y) {
                for (int x=0; x<8; ++x) {
                    std::cout << quat_gjpegencoder[idx++] << " ";
                }
                std::cout << "\n";
            }

            for (int i=0; i<64; ++i) {
                int idx = m*64+i;
                short q0 = quat_gpujpeg[idx];
                short q1 = quat_gjpegencoder[idx];
                if (q0 != q1) {
                    std::cerr << "m: " << m << ", id: " << i << ", quat: "<< q0 << "/" << q1 << std::endl;
                    ++num;
                }
            }
        }
        
    }
}

void test() {
    const int WARP_COUNT = 4;

    typedef float dct_t;

    // dimensions of shared buffer (compile time constants)
    enum {
        // 4 8x8 blocks, padded to odd number of 4byte banks
        SHARED_STRIDE = ((32 * sizeof(dct_t)) | 4) / sizeof(dct_t),

        // number of shared buffer items needed for 1 warp
        SHARED_SIZE_WARP = SHARED_STRIDE * 8,

        // total number of items in shared buffer
        SHARED_SIZE_TOTAL = SHARED_SIZE_WARP * WARP_COUNT
    };

    int x = SHARED_STRIDE;
    int y = SHARED_SIZE_WARP;
    int z = SHARED_SIZE_TOTAL;



    std::cout << "ok";
}

int main(int argc, char* argv[]) {

   
    test();


    


    // std::string quat_gpujpeg_f = "/home/wangrui22/projects/mi-jpeg/bin/gpujpeg-quat-1-0.quat";
    // unsigned int len = 64;
    // short* quat_gpujpeg = new short[len];

    // {
    //     std::ifstream in(quat_gpujpeg_f, std::ios::binary | std::ios::in);
    //     if (!in.is_open()) {
    //         return -1;
    //     }
    //     in.read((char*)quat_gpujpeg, len*2);
    //     in.close();
    // }

    // short quat111[64];
    // for (int i=0; i<64; ++i) {
    //     quat111[i] = quat_gpujpeg[i];
    // }


    // int width = 512;
    // int height = 512;
    // unsigned char* gray = new unsigned char[width*height];
    // std::string file_path = "/home/wangrui22/projects/mi-jpeg/data/gray3-512-512.raw";
    // std::ifstream in(file_path, std::ios::binary | std::ios::in);
    // if (!in.is_open()) {
    //     return -1;
    // }
    // in.read((char*)(gray), width*height);
    // in.close();

    // unsigned char* rgb = new unsigned char[width*height*3];
    // for (int i=0; i<width*height; ++i) {
    //     rgb[i*3] = gray[i];
    //     rgb[i*3+1] = gray[i];
    //     rgb[i*3+2] = gray[i];
    // }

    // {
    //     std::ofstream out("/home/wangrui22/projects/mi-jpeg/data/gray3-512-512-rgb.raw", std::ios::out|std::ios::binary);
    //     if (out.is_open()) {
    //         out.write((char*)rgb, width*height*3);
    //         out.close();
    //         //return 0;
    //     }
    // }
    

    check_yuv_and_quat();

    std::cout << "done\n";
    

    return 0;
}
