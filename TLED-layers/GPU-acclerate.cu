#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>


#define TILE_SIZE 16
#define BLOCK_SIZE 256

using namespace std;

#define APPRO_MUL

#define LENGTH 8

texture<int, cudaTextureType2D, cudaReadModeElementType> lookupTable;
cudaArray* cuArray;

void setupTexture(int width, int height, int* h_data){
    cudaMallocArray(&cuArray, &lookupTable.channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, h_data, width * height * sizeof(float), cudaMemcpyHostToDevice);

    lookupTable.addressMode[0] = cudaAddressModeWrap;
    lookupTable.addressMode[1] = cudaAddressModeWrap;
    lookupTable.filterMode = cudaFilterModePoint;
    lookupTable.normalized = false;

    cudaBindTextureToArray(lookupTable, cuArray);
}

void cleanupTexture(){
    cudaUnbindTexture(lookupTable);
    cudaFreeArray(cuArray);
}

__device__ void testTexture(){
    printf("DEBUG: test the texture memory: %d\n", tex2D(lookupTable, 78, 56));
}

__global__ void testTextureKernel(){
    testTexture();
}

__device__ int float_to_fix_gpu(float floatNumber, float scale){
    return abs(int(floatNumber / scale));
}

__device__ float Approxiamte_multiplication_gpu(float x, float y, float x_max, float y_max, int t){
    float x_scale = x_max / ((1 << (LENGTH-1))-1); // LENGTH = 8, 255
    float y_scale = y_max / ((1 << (LENGTH-1))-1);
    int x_fixed = float_to_fix_gpu(x, x_scale);
    int y_fixed = float_to_fix_gpu(y, y_scale);
    if((abs(x_fixed) < 1e-8) || (abs(y_fixed) < 1e-8))
        return 0.0;
    else if(t <= 1e-5){
        return x*y;
    }
    else{
        int sign =  1 - 2*((x>0)^(y>0));
        float sum = 0.0;
        unsigned int pos = 1;
        for (int i = 0;i < t;i++) {
            if (y_fixed & (pos << i)) {
                sum += (x_fixed >> (t - i)) << t;
            }
        }
        for (int i = t;i < LENGTH;i++) {
            if (y_fixed & (pos << i)) {
                sum += x_fixed << i;
            }
        }
        sum = sum * x_scale * y_scale;
        return sign * sum;
    }
}

__device__ float Approxiamte_multiplication_int_gpu(int x, int y, float x_scale, float y_scale, int x_zero, int y_zero, int t){

    if((abs(x) < 1e-8) || (abs(y) < 1e-8))
        return 0.0;
    else if(t <= 1e-5){
        return x*y;
    }
    else{
        float sum = 0.0; //q_1*q_2
        unsigned int pos = 1;
        for (int i = 0;i < t;i++) {
            if (y & (pos << i)) {
                sum += (x >> (t - i)) << t;
            }
        }
        for (int i = t;i < LENGTH;i++) {
            if (y & (pos << i)) {
                sum += x << i;
            }
        }
        return x_scale * y_scale * (sum + x_zero * y_zero - x_zero * y - y_zero * x);
    }
}

__device__ float Approxiamte_multiplication_gpu_any(float x, float y, float x_max, float y_max){
    float x_scale = x_max / ((1 << (LENGTH-1))-1); // LENGTH = 8, 255
    float y_scale = y_max / ((1 << (LENGTH-1))-1);
    int x_fixed = float_to_fix_gpu(x, x_scale);
    int y_fixed = float_to_fix_gpu(y, y_scale);
    int sign =  1 - 2*((x>0)^(y>0));
    return sign * tex2D(lookupTable, x_fixed, y_fixed) * x_scale * y_scale;
}


__global__ void operator_matmul_h(const float *input1, const float *input2, float *output, int height, int k, int width, int broadcast, float x_max, float y_max, int approximate) {
    __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

    int batch_idx = blockIdx.z;
    if (broadcast != 1) input1 += batch_idx * height * k;
    if (broadcast != 2) input2 += batch_idx * k * width;
    output += batch_idx * height * width;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    float v = 0;

    for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
        if (i * TILE_SIZE + ty < k && row < height)
        shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
        else
        shared_input1[tx][ty] = 0;

        if (i * TILE_SIZE + tx < k && col < width)
            shared_input2[tx][ty] = input2[(i * TILE_SIZE + tx) * width + col];
        else
            shared_input2[tx][ty] = 0;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
        #ifdef APPRO_MUL
            v += Approxiamte_multiplication_gpu(shared_input1[tx][j], shared_input2[j][ty], x_max, y_max, approximate);
        #else
            v += shared_input1[tx][j] * shared_input2[j][ty];
        #endif
        __syncthreads();
    }
    if (row < height && col < width) output[row * width + col] = v;
}

__global__ void operator_matmul_h_col(const float *input1, const float *input2, float *output, int height, int k, int width, int broadcast, float x_max, float y_max, int approximate) {
    __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

    int batch_idx = blockIdx.z;
    if (broadcast != 1) input1 += batch_idx * height * k;
    if (broadcast != 2) input2 += batch_idx * width * k; 
    output += batch_idx * height * width;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    float v = 0;

    for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
        if (i * TILE_SIZE + ty < k && row < height)
            shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * TILE_SIZE + tx < k && col < width)
            shared_input2[tx][ty] = input2[col * k + i * TILE_SIZE + tx];
        else
            shared_input2[tx][ty] = 0;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
            #ifdef APPRO_MUL
                v += Approxiamte_multiplication_gpu(shared_input1[tx][j], shared_input2[j][ty], x_max, y_max, approximate);
            #else
                v += shared_input1[tx][j] * shared_input2[j][ty];
            #endif
        __syncthreads();
    }

    if (row < height && col < width) output[row * width + col] = v;
}

__global__ void operator_matmul_h_col_int(const int* input1, const float scale_input1, const int zero_input1, const int *input2, const float scale_input2, const float zero_input2, float *output, int height, int k, int width, int broadcast, int approximate) {
    __shared__ int shared_input1[TILE_SIZE][TILE_SIZE];
    __shared__ int shared_input2[TILE_SIZE][TILE_SIZE];

    int batch_idx = blockIdx.z;
    if (broadcast != 1) input1 += batch_idx * height * k;
    if (broadcast != 2) input2 += batch_idx * width * k; 
    output += batch_idx * height * width;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    float v = 0;

    for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
        if (i * TILE_SIZE + ty < k && row < height)
            shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * TILE_SIZE + tx < k && col < width)
            shared_input2[tx][ty] = input2[col * k + i * TILE_SIZE + tx]; 
        else
            shared_input2[tx][ty] = 0;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
            #ifdef APPRO_MUL
                v += Approxiamte_multiplication_int_gpu(shared_input1[tx][j], shared_input2[j][ty], scale_input1, scale_input2, zero_input1, zero_input2, approximate);
            #else
                v += shared_input1[tx][j] * shared_input2[j][ty];
            #endif
        __syncthreads();
    }

    if (row < height && col < width) output[row * width + col] = v;
}

__global__ void operator_matmul_h_col_any(const float *input1, const float *input2, float *output, int height, int k, int width, int broadcast, float x_max, float y_max) {
    __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

    int batch_idx = blockIdx.z;
    if (broadcast != 1) input1 += batch_idx * height * k;
    if (broadcast != 2) input2 += batch_idx * width * k;
    output += batch_idx * height * width;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    float v = 0;

    for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
        if (i * TILE_SIZE + ty < k && row < height)
            shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * TILE_SIZE + tx < k && col < width)
            shared_input2[tx][ty] = input2[col * k + i * TILE_SIZE + tx];
        else
            shared_input2[tx][ty] = 0;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
            #ifdef APPRO_MUL
                v += Approxiamte_multiplication_gpu_any(shared_input1[tx][j], shared_input2[j][ty], x_max, y_max);
            #else
                v += shared_input1[tx][j] * shared_input2[j][ty];
            #endif
        __syncthreads();
    }

    if (row < height && col < width) output[row * width + col] = v;
}

__global__ void operator_matmul_h_any(const float *input1, const float *input2, float *output, int height, int k, int width, int broadcast, float x_max, float y_max) {
    __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

    int batch_idx = blockIdx.z;
    if (broadcast != 1) input1 += batch_idx * height * k;
    if (broadcast != 2) input2 += batch_idx * k * width;
    output += batch_idx * height * width;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    float v = 0;

    for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
        if (i * TILE_SIZE + ty < k && row < height)
        shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
        else
        shared_input1[tx][ty] = 0;

        if (i * TILE_SIZE + tx < k && col < width)
        shared_input2[tx][ty] = input2[(i * TILE_SIZE + tx) * width + col];
        else
        shared_input2[tx][ty] = 0;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
        #ifdef APPRO_MUL
            v += Approxiamte_multiplication_gpu_any(shared_input1[tx][j], shared_input2[j][ty], x_max, y_max);
        #else
            v += shared_input1[tx][j] * shared_input2[j][ty];
        #endif
        __syncthreads();
    }
    if (row < height && col < width) output[row * width + col] = v;
}

__device__ float Approximate_multiplication_derivative_one_gpu(float x, float y, float x_max, float y_max, int t){

    int sign =  1 - 2*((x>0)^(y>0));
    float x_scale = x_max / ((1 << (LENGTH-1))-1); // LENGTH = 8, 255
    float y_scale = y_max / ((1 << (LENGTH-1))-1);
    int x_fixed = float_to_fix_gpu(x, x_scale);
    int y_fixed = float_to_fix_gpu(y, y_scale);
    float grad = 0.0;
    unsigned int pos = 1;
    // calculate sum of the power of t+1
    for(int i=0; i<t+2; i++){
        if (x_fixed & (pos << i)) {
            grad += (y_fixed & (pos << (t+1-i)) ) << i;
        }
    }
    return sign * grad * x_scale * y_scale;;
}

__global__ void ApproximateMultiplicationDerivativeKernel(const float *weights, const float *x, float *output, int height, int k, int width, float x_max, float y_max, int approximate){ 
    const int width_idx = blockIdx.x;
    if(width_idx < width){
        float sum = 0;
        for(int i=0; i<height; i++){
            for(int j=0; j<k; j++){
                assert(j*width+width_idx < k*width);
                assert(i*k+j < height*k);
                sum += Approximate_multiplication_derivative_one_gpu(weights[j*width+width_idx], x[i*k+j], x_max, y_max, approximate);
            }
        }
        output[width_idx] = sum / height;
    }

}

__global__ void im2col_h(const int n, const float *data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float *data_col, int im_stride, int col_stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        const int batch_idx = blockIdx.y;
        data_im += batch_idx * im_stride;
        data_col += batch_idx * col_stride;

        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;

        // channel offset
        float *data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const float *data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        // copy to col
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

__global__ void ApproximateCovolutionDerivativeKernel(const float* image, const float* kernel, const float *delta, float *output, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels, const int height_col, const int width_col, const float x_max, const float y_max, const int approximate){
    int in_channels_idx = blockIdx.x;
    int out_channels_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int kerner_idx = threadIdx.x;
    int kx = kerner_idx / kernel_size;
    int ky = kerner_idx % kernel_size;
    double sum_value = 0;
    // for each output pixel
    for(int x=0; x<height_col; ++x){
        for(int y=0; y<width_col; ++y){
            if(x+kx < padding_h || y+ky < padding_w || x+kx >= height+padding_h || y+ky >= width+padding_w)
                sum_value += 0.0;
            else{
                sum_value += delta[batch_idx * (out_channels*height_col*width_col) + out_channels_idx*(height_col*width_col) + x*(width_col) + y] * Approximate_multiplication_derivative_one_gpu(image[batch_idx * (in_channels*height*width) + in_channels_idx*(height*width) + (stride*x+kx-padding_h) * width + stride*y + ky -padding_w], kernel[out_channels_idx*in_channels*kernel_size*kernel_size+in_channels_idx*kernel_size*kernel_size+kx*kernel_size+ky], x_max, y_max, approximate);
            }
        }
    }
    atomicAdd(output, sum_value / batch_size);
}

void Approximate_Dot(const float* input1, const float* input2, float* output, int height, int k, int width, float x_max, float y_max, int approximate) {
    float *input1_ptr, *input2_ptr, *output_ptr;

    cudaMallocManaged((void**)&input1_ptr, height * k * sizeof(float));
    cudaMallocManaged((void**)&input2_ptr, k * width * sizeof(float));
    cudaMallocManaged((void**)&output_ptr, height * width * sizeof(float));
    cudaMemcpy(input1_ptr, input1, height * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input2_ptr, input2, k * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid(ceil((float) width/ TILE_SIZE), ceil((float)height / TILE_SIZE), 1);
    operator_matmul_h_col<<<dim_grid, dim_block>>>(input1_ptr, input2_ptr, output_ptr, height, k, width, 1, x_max, y_max, approximate);
    cudaMemcpy(output, output_ptr, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    cudaFree(input1_ptr);   
    cudaFree(input2_ptr);
    cudaFree(output_ptr);
}

void Approximate_multiplication_derivative_GPU(const float *weights, const float *x, float *out_grad, int height, int k, int width, float x_max, float y_max, int approximate){

    dim3 dim_grid(width);
    ApproximateMultiplicationDerivativeKernel<<<dim_grid, 1>>>(weights, x, out_grad, height, k, width, x_max, y_max, approximate);
    cudaDeviceSynchronize();
}

void Approximate_convolution(const float* image, const float* kernel, float *output, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels, const float x_max, const float y_max, const int approximate){
    int height_col = (height + 2 * padding_h - kernel_size) / stride + 1;
    int width_col = (width + 2 * padding_w - kernel_size) / stride + 1;
    int size = in_channels * height_col * width_col;

    int im_stride = in_channels * height * width;
    int col_stride = in_channels * kernel_size * kernel_size * height_col * width_col;

    float *data_im, *data_col;
    cudaMallocManaged((void**)&data_im, batch_size * in_channels * height * width * sizeof(float));
    cudaMallocManaged((void**)&data_col, batch_size * in_channels * kernel_size * kernel_size * height_col * width_col * sizeof(float));
    cudaMemcpy(data_im, image, batch_size * in_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_grid(ceil((float)size / BLOCK_SIZE), batch_size);
    im2col_h<<<dim_grid, BLOCK_SIZE>>>(size, data_im, height, width, kernel_size, kernel_size, padding_h, padding_w, stride, stride, height_col, width_col, data_col, im_stride, col_stride);

    float *kernel_ptr, *output_ptr;
    cudaMallocManaged((void**)&kernel_ptr, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMallocManaged((void**)&output_ptr, batch_size * out_channels * height_col * width_col * sizeof(float));
    cudaMemcpy(kernel_ptr, kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_SIZE, TILE_SIZE);

    dim3 dim_grid_(ceil((float) (height_col * width_col)/ TILE_SIZE), ceil((float)out_channels / TILE_SIZE), batch_size);

    operator_matmul_h<<<dim_grid_, dim_block>>>(kernel_ptr, data_col, output_ptr, out_channels, in_channels * kernel_size * kernel_size, height_col * width_col, 1, y_max, x_max, approximate);
    cudaMemcpy(output, output_ptr, batch_size * out_channels * height_col * width_col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(data_im);
    cudaFree(data_col);
    cudaFree(kernel_ptr);
    cudaFree(output_ptr);
    
}

void Convolution_Parameter_Gradient_GPU(const float *image, const float *kernel, const float *delta, float *approximate_grad, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels, const int height_col, const int width_col, const float x_max, const float y_max, const int approximate){
    float *image_ptr, *kernel_ptr, *approximate_grad_ptr, *delta_ptr;
    cudaMallocManaged((void**)&image_ptr, batch_size * in_channels * height * width * sizeof(float));
    cudaMallocManaged((void**)&kernel_ptr, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMallocManaged((void**)&approximate_grad_ptr, sizeof(float));
    cudaMallocManaged((void**)&delta_ptr, batch_size * out_channels * height_col * width_col * sizeof(float));
    cudaMemcpy(image_ptr, image, batch_size * in_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_ptr, kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_ptr, delta, batch_size * out_channels * height_col * width_col * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_grid(in_channels, out_channels, batch_size);
    ApproximateCovolutionDerivativeKernel<<<dim_grid, kernel_size*kernel_size>>>(image_ptr, kernel_ptr, delta_ptr, approximate_grad_ptr, stride, padding_h, padding_w, kernel_size, batch_size, height, width, in_channels, out_channels, height_col, width_col, x_max, y_max, approximate);
    cudaMemcpy(approximate_grad, approximate_grad_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(image_ptr);
    cudaFree(kernel_ptr);
    cudaFree(approximate_grad_ptr);
    cudaFree(delta_ptr);
    
}

void Approximate_INTDot(const int* input1, const float scale_input1, const int zero_input1, const int *input2, const float scale_input2, const float zero_input2, float *output, int height, int k, int width, const int approximate){
    int *input1_ptr, *input2_ptr;
    float *output_ptr;
    cudaMallocManaged((void**)&input1_ptr, height * k * sizeof(int));
    cudaMallocManaged((void**)&input2_ptr, k * width * sizeof(int));
    cudaMallocManaged((void**)&output_ptr, height * width * sizeof(float));
    cudaMemcpy(input1_ptr, input1, height * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(input2_ptr, input2, k * width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid(ceil((float) width/ TILE_SIZE), ceil((float)height / TILE_SIZE), 1);
    operator_matmul_h_col_int<<<dim_grid, dim_block>>>(input1_ptr, scale_input1, zero_input1, input2_ptr, scale_input2, zero_input2, output_ptr, height, k, width, 1, approximate);
    cudaMemcpy(output, output_ptr, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(input1_ptr);   
    cudaFree(input2_ptr);
    cudaFree(output_ptr);
}

void Approximate_AnyDot(const float* input1, const float* input2, float* output, int height, int k, int width, float x_max, float y_max, int *Approximate_LUT){
    setupTexture(256, 256, Approximate_LUT);
    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid(ceil((float) width/ TILE_SIZE), ceil((float)height / TILE_SIZE), 1);
    operator_matmul_h_col_any<<<dim_grid, dim_block>>>(input1, input2, output, height, k, width, 1, x_max, y_max);
    cleanupTexture();
}

void Approximate_Anyconvolution(const float* image, const float* kernel, float *output, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels, const float x_max, const float y_max, int *Approximate_LUT){
    int height_col = (height + 2 * padding_h - kernel_size) / stride + 1;
    int width_col = (width + 2 * padding_w - kernel_size) / stride + 1;
    int size = in_channels * height_col * width_col;

    int im_stride = in_channels * height * width;
    int col_stride = in_channels * kernel_size * kernel_size * height_col * width_col;

    setupTexture(256, 256, Approximate_LUT);
    float *data_col;
    cudaMallocManaged((void**)&data_col, batch_size * in_channels * kernel_size * kernel_size * height_col * width_col * sizeof(float));

    dim3 dim_grid(ceil((float)size / BLOCK_SIZE), batch_size);
    im2col_h<<<dim_grid, BLOCK_SIZE>>>(size, image, height, width, kernel_size, kernel_size, padding_h, padding_w, stride, stride, height_col, width_col, data_col, im_stride, col_stride);

    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid_(ceil((float) (height_col * width_col)/ TILE_SIZE), ceil((float)out_channels / TILE_SIZE), batch_size);
    operator_matmul_h_any<<<dim_grid_, dim_block>>>(kernel, data_col, output, out_channels, in_channels * kernel_size * kernel_size, height_col * width_col, 1, y_max, x_max);

    cudaDeviceSynchronize();
    cudaFree(data_col);
    cleanupTexture();
}



