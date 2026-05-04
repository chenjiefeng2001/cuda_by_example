// heat_2d

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"
#include "../../common/cpu_anim.h"

#define DIM 1024
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// ==========================================
// 结构体声明
// ==========================================
struct DataBlock
{
    unsigned char* dev_bitmap;
    float* dev_inSrc;       // 输入缓冲区
    float* dev_outSrc;      // 输出缓冲区
    float* dev_constSrc;    // 初始化的热源
    CPUAnimBitmap* bitmap;

    cudaEvent_t start, stop;
    float totalTime;
    float frames;

    cudaTextureObject_t texConstSrcObj;
    cudaTextureObject_t texInObj;
    cudaTextureObject_t texOutObj;
};

// ==========================================
// 防止 cpu_anim.h 底层鼠标点击触发空指针异常的兜底函数
// ==========================================
void dummy_click(void* data, int x, int y, int tx, int ty) {
    // 什么都不做，仅用于防止空指针调用崩溃
}

// ==========================================
// 核函数定义
// ==========================================
__global__ void copy_const_kernel(float* iptr, cudaTextureObject_t texConstSrc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    float center = tex2D<float>(texConstSrc, x, y);
    if (center != 0)
        iptr[offset] = center;
}

__global__ void blend_kernel(float* dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    float t, l, c, r, b;
    if (dstOut) {
        t = tex2D<float>(texIn, x, y - 1); // top
        l = tex2D<float>(texIn, x - 1, y); // left
        c = tex2D<float>(texIn, x, y);     // center
        r = tex2D<float>(texIn, x + 1, y); // right
        b = tex2D<float>(texIn, x, y + 1); // bottom
    }
    else {
        t = tex2D<float>(texOut, x, y - 1);
        l = tex2D<float>(texOut, x - 1, y);
        c = tex2D<float>(texOut, x, y);
        r = tex2D<float>(texOut, x + 1, y);
        b = tex2D<float>(texOut, x, y + 1);
    }
    dst[offset] = c + SPEED * (t + b + l + r - 4 * c);
}

// ==========================================
// 创建纹理对象的辅助函数（增加了更严格的配置）
// ==========================================
cudaTextureObject_t createTextureObject(float* devPtr, int width, int height) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devPtr;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    // 【重要强化】：对于2D纹理，务必设置 Clamp，防止越界读取导致驱动崩溃
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0; // 不使用归一化坐标

    cudaTextureObject_t texObj = 0;
    HANDLE_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}

// ==========================================
// 每一帧动画将调用的函数
// ==========================================
void anim_gpu(DataBlock* data, int ticks) {
    HANDLE_ERROR(cudaEventRecord(data->start, 0));

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap* bitmap = data->bitmap;

    volatile bool dstOut = true;
    for (int i = 0; i < 90; i++) {
        float* in, * out;
        if (dstOut) {
            in = data->dev_inSrc;
            out = data->dev_outSrc;
        }
        else {
            in = data->dev_outSrc;
            out = data->dev_inSrc;
        }

        copy_const_kernel << <blocks, threads >> > (in, data->texConstSrcObj);
        blend_kernel << <blocks, threads >> > (out, dstOut, data->texInObj, data->texOutObj);

        dstOut = !dstOut;
    }

    // book.h自带的 float_to_color
    float_to_color << <blocks, threads >> > (data->dev_bitmap, data->dev_inSrc);

    // 将结果复制回CPU用于显示
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
        data->dev_bitmap,
        bitmap->image_size(),
        cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(data->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(data->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, data->start, data->stop));

    data->totalTime += elapsedTime;
    data->frames++;
    printf("Average Time per frame:  %3.1f ms\n", data->totalTime / data->frames);
}

// 退出时的清理工作
void anim_exit(DataBlock* data) {
    HANDLE_ERROR(cudaDestroyTextureObject(data->texConstSrcObj));
    HANDLE_ERROR(cudaDestroyTextureObject(data->texInObj));
    HANDLE_ERROR(cudaDestroyTextureObject(data->texOutObj));

    HANDLE_ERROR(cudaFree(data->dev_inSrc));
    HANDLE_ERROR(cudaFree(data->dev_outSrc));
    HANDLE_ERROR(cudaFree(data->dev_constSrc));
    // 修复原版中的一个小内存泄漏
    HANDLE_ERROR(cudaFree(data->dev_bitmap));

    HANDLE_ERROR(cudaEventDestroy(data->start));
    HANDLE_ERROR(cudaEventDestroy(data->stop));
}

int main() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;

    // 【关键修复点】：绑定一个空的点击事件，防止你在画面上点击鼠标导致 0x000000 崩溃
    bitmap.clickDrag = dummy_click;

    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));

    // 创建纹理对象
    data.texConstSrcObj = createTextureObject(data.dev_constSrc, DIM, DIM);
    data.texInObj = createTextureObject(data.dev_inSrc, DIM, DIM);
    data.texOutObj = createTextureObject(data.dev_outSrc, DIM, DIM);

    float* temp = (float*)malloc(bitmap.image_size());

    // 初始化热源
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
    free(temp);

    // 开始动画循环
    bitmap.anim_and_exit((void (*)(void*, int))anim_gpu, (void (*)(void*))anim_exit);

    return 0;
}