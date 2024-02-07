#include <torch/extension.h>
#include "tled_layers.h"

void torch_Approximate_Dot(const torch::Tensor &input1, const torch::Tensor &input2, torch::Tensor &output, int height, int k, int width, int approximate){
    torch::Tensor abs_input1 = input1.abs();
    torch::Tensor abs_input2 = input2.abs();
    float x_max = abs_input1.max().item<float>();
    float y_max = abs_input2.max().item<float>();

    Approximate_Dot((const float *)input1.data_ptr(),
                (const float *)input2.data_ptr(),
                (float *)output.data_ptr(),
                height, k, width, x_max, y_max, approximate);
}

void torch_Approximate_multiplication_derivative_GPU(const torch::Tensor &weights, const torch::Tensor &x, torch::Tensor &out_grad, int height, int k, int width, int approximate){
    torch::Tensor abs_weights = weights.abs();
    torch::Tensor abs_x = x.abs();
    float x_max = abs_weights.max().item<float>();
    float y_max = abs_x.max().item<float>();
    Approximate_multiplication_derivative_GPU((const float *)weights.data_ptr(),
                (const float *)x.data_ptr(),
                (float *)out_grad.data_ptr(),
                height, k, width, x_max, y_max, approximate);
}

void torch_Approximate_convolution(const torch::Tensor &image, const torch::Tensor &kernel, torch::Tensor &output, const int stride, const int padding_h, const int padding_w, const int approximate){
    // const float* image, const float* kernel, float *output, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels
    const int batch_size = image.size(0);
    const int in_channels = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    const int out_channels = kernel.size(0);
    assert(kernel.size(1) == in_channels);
    const int kernel_size = kernel.size(2);
    assert(kernel.size(3) == kernel_size);

    torch::Tensor abs_image = image.abs();
    torch::Tensor abs_kernel = kernel.abs();

    float x_max = abs_image.max().item<float>();
    float y_max = abs_kernel.max().item<float>();
    // Approximate_convolution(const float* image, const float* kernel, float *output, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels, const int approximate, const int fraction_bits)
    Approximate_convolution((const float *)image.data_ptr(), (const float *)kernel.data_ptr(), (float *)output.data_ptr(), stride, padding_h, padding_w, kernel_size, batch_size, height, width, in_channels, out_channels, x_max, y_max, approximate);
}

void torch_Convolution_Parameter_Gradient_GPU(const torch::Tensor &image, const torch::Tensor &kernel, const torch::Tensor &delta, torch::Tensor &out_grad, const int stride, const int padding_h, const int padding_w, const int approximate){
    const int batch_size = image.size(0);
    const int in_channels = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    const int out_channels = kernel.size(0);
    assert(kernel.size(1) == in_channels);
    const int kernel_size = kernel.size(2);
    assert(kernel.size(3) == kernel_size);

    const int height_col = delta.size(2);
    const int width_col = delta.size(3);

    torch::Tensor abs_image = image.abs();
    torch::Tensor abs_kernel = kernel.abs();
    
    float x_max = abs_image.max().item<float>();
    float y_max = abs_kernel.max().item<float>();

    Convolution_Parameter_Gradient_GPU((const float *)image.data_ptr(), (const float *)kernel.data_ptr(), (const float *)delta.data_ptr(), (float *)out_grad.data_ptr(), stride, padding_h, padding_w, kernel_size, batch_size, height, width, in_channels, out_channels, height_col, width_col, x_max, y_max, approximate);
}

void torch_Approximate_AnyDot(const torch::Tensor &input1, const torch::Tensor &input2, torch::Tensor &output, int height, int k, int width, torch::Tensor &Approximate_LUT){
    torch::Tensor abs_input1 = input1.abs();
    torch::Tensor abs_input2 = input2.abs();
    float x_max = abs_input1.max().item<float>();
    float y_max = abs_input2.max().item<float>();

    Approximate_AnyDot((const float *)input1.data_ptr(),
                (const float *)input2.data_ptr(),
                (float *)output.data_ptr(),
                height, k, width, x_max, y_max, (int *)Approximate_LUT.data_ptr());
}

void torch_Approximate_Anyconvolution(const torch::Tensor &image, const torch::Tensor &kernel, torch::Tensor &output, const int stride, const int padding_h, const int padding_w, torch::Tensor &Approximate_LUT){
    // const float* image, const float* kernel, float *output, const int stride, const int padding_h, const int padding_w, const int kernel_size, const int batch_size, const int height, const int width, const int in_channels, const int out_channels
    const int batch_size = image.size(0);
    const int in_channels = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    const int out_channels = kernel.size(0);
    assert(kernel.size(1) == in_channels);
    const int kernel_size = kernel.size(2);
    assert(kernel.size(3) == kernel_size);

    torch::Tensor abs_image = image.abs();
    torch::Tensor abs_kernel = kernel.abs();

    float x_max = abs_image.max().item<float>();
    float y_max = abs_kernel.max().item<float>();

    Approximate_Anyconvolution((const float *)image.data_ptr(), (const float *)kernel.data_ptr(), (float *)output.data_ptr(), stride, padding_h, padding_w, kernel_size, batch_size, height, width, in_channels, out_channels, x_max, y_max, (int *)Approximate_LUT.data_ptr());

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_Approximate_Dot",
          &torch_Approximate_Dot,
          "Approximate_Dot kernel warpper");
    m.def("torch_Approximate_multiplication_derivative_GPU",
          &torch_Approximate_multiplication_derivative_GPU,
          "Approximate_multiplication_derivative kernel warpper");
    m.def("torch_Approximate_convolution",
            &torch_Approximate_convolution,
            "Approximate_convolution kernel warpper");
    m.def("torch_Convolution_Parameter_Gradient_GPU",
            &torch_Convolution_Parameter_Gradient_GPU,
            "Convolution_Parameter_Gradient_GPU kernel warpper");
    m.def("torch_Approximate_AnyDot",
            &torch_Approximate_AnyDot,
            "Approximate_AnyDot kernel warpper");
    m.def("torch_Approximate_Anyconvolution",
            &torch_Approximate_Anyconvolution,
            "Approximate_Anyconvolution kernel warpper");
}