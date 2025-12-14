#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

__device__ void color_pixel(
    const int iter,
    const int maxIter,
    const double zx,
    const double zy,
    uint8_t *pixel
) {
    if (iter == maxIter) {
        pixel[0] = pixel[1] = pixel[2] = 0;
    } else {
        const double smooth = iter - log2(log2(zx * zx + zy * zy));
        const double hue = 360.0 * smooth / maxIter;
        const double c = hue / 360.0 * 6.0;
        const double x1 = c - floor(c);

        double r = 0, g = 0, b = 0;

        // evil color magic
        if (const int i = static_cast<int>(floor(c)); i == 0) {
            r = 1;
            g = x1;
            b = 0;
        } else if (i == 1) {
            r = 1 - x1;
            g = 1;
            b = 0;
        } else if (i == 2) {
            r = 0;
            g = 1;
            b = x1;
        } else if (i == 3) {
            r = 0;
            g = 1 - x1;
            b = 1;
        } else if (i == 4) {
            r = x1;
            g = 0;
            b = 1;
        } else {
            r = 1;
            g = 0;
            b = 1 - x1;
        }

        pixel[0] = static_cast<uint8_t>(r * 255);
        pixel[1] = static_cast<uint8_t>(g * 255);
        pixel[2] = static_cast<uint8_t>(b * 255);
    }
    pixel[3] = 255;
}

__global__ void mandelbrot_kernel(
    uint8_t *pixels,
    const int width,
    const int height,
    const double centerX,
    const double centerY,
    const double scale,
    const int maxIter
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const double cx = centerX + (x - width / 2.0) * scale / width;
    const double cy = centerY + (y - height / 2.0) * scale / width;

    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
        const double tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        iter++;
    }

    const int idx = (y * width + x) * 4;
    color_pixel(iter, maxIter, zx, zy, &pixels[idx]);
}

void mandelbrot_cuda(
    uint8_t *pixels_d,
    const int width,
    const int height,
    const double centerX,
    const double centerY,
    const double scale,
    const int maxIter
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    mandelbrot_kernel<<<grid, block>>>(pixels_d, width, height, centerX, centerY, scale, maxIter);

    cudaDeviceSynchronize();
}
