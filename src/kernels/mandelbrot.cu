#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

/**
 * @brief Colors a pixel based on the number of iterations and the final Z value.
 *
 * Uses a smooth coloring algorithm to determine the RGB values.
 *
 * @param iter The number of iterations reached.
 * @param maxIter The maximum number of iterations allowed.
 * @param zx The real part of the final Z value.
 * @param zy The imaginary part of the final Z value.
 * @param pixel Pointer to the pixel's RGBA values.
 * @param theme The color theme to use.
 */
__device__ void color_pixel(
    const int iter,
    const int maxIter,
    const double zx,
    const double zy,
    uint8_t *pixel,
    const int theme
) {
    if (iter == maxIter) {
        pixel[0] = pixel[1] = pixel[2] = 0;
    } else {
        const double smooth = iter - log2(log2(zx * zx + zy * zy));

        if (theme == 1) {
            // BLUE_GOLD
            const double t = smooth * 0.15;
            pixel[0] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(3.0 + t + 0.0)));
            pixel[1] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(3.0 + t + 0.6)));
            pixel[2] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(3.0 + t + 1.0)));
        } else if (theme == 3) {
            // NEON
            const double t = smooth * 0.2;
            pixel[0] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(t)));
            pixel[1] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(t + 2.0)));
            pixel[2] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(t + 4.0)));
        } else if (theme == 4) {
            // FIRE - red/orange/yellow flames
            const double t = smooth * 0.1;
            const double r = 255 * (1.5 - 0.5 * cos(t));
            const double b = 0.2 * cos(t * 2.0);
            pixel[0] = static_cast<uint8_t>(r > 255 ? 255 : r);
            pixel[1] = static_cast<uint8_t>(255 * (0.3 + 0.4 * cos(t + 0.5)));
            pixel[2] = static_cast<uint8_t>(255 * (b > 0.0 ? b : 0.0));
        } else if (theme == 5) {
            // OCEAN - deep blue to cyan
            const double t = smooth * 0.12;
            pixel[0] = static_cast<uint8_t>(255 * (0.1 + 0.2 * cos(t + 2.0)));
            pixel[1] = static_cast<uint8_t>(255 * (0.3 + 0.4 * cos(t + 1.0)));
            pixel[2] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(t)));
        } else if (theme == 6) {
            // PURPLE - purple/pink
            const double t = smooth * 0.15;
            pixel[0] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(t)));
            pixel[1] = static_cast<uint8_t>(255 * (0.2 + 0.3 * cos(t + 2.5)));
            pixel[2] = static_cast<uint8_t>(255 * (0.5 + 0.5 * cos(t + 1.0)));
        } else if (theme == 7) {
            // GRAYSCALE - classic black and white
            const double intensity = 0.5 + 0.5 * cos(smooth * 0.1);
            const auto gray = static_cast<uint8_t>(255 * intensity);
            pixel[0] = pixel[1] = pixel[2] = gray;
        } else if (theme == 8) {
            // ELECTRIC - electric blue/white high contrast
            const double t = smooth * 0.25;
            const double intensity = pow(0.5 + 0.5 * cos(t), 2.0);
            pixel[0] = static_cast<uint8_t>(255 * (0.3 + 0.7 * intensity));
            pixel[1] = static_cast<uint8_t>(255 * (0.5 + 0.5 * intensity));
            pixel[2] = static_cast<uint8_t>(255 * (0.8 + 0.2 * intensity));
        } else {
            // DEFAULT (Rainbow)
            const double hue = 360.0 * smooth / maxIter;
            const double c = hue / 360.0 * 6.0;
            const double x1 = c - floor(c);
            double r = 0, g = 0, b = 0;
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
    }
    pixel[3] = 255;
}

/**
 * @brief CUDA kernel to compute the Mandelbrot set.
 *
 * Computes the Mandelbrot set for a given region and stores the result in the pixels array.
 *
 * @param pixels Pointer to the output pixel array (RGBA).
 * @param width Width of the image.
 * @param height Height of the image.
 * @param centerX X coordinate of the center of the view.
 * @param centerY Y coordinate of the center of the view.
 * @param scale Scale of the view (width in complex plane units).
 * @param maxIter Maximum number of iterations.
 * @param theme The color theme to use.
 */
__global__ void mandelbrot_kernel(
    uint8_t *pixels,
    const int width,
    const int height,
    const double centerX,
    const double centerY,
    const double scale,
    const int maxIter,
    const int theme
) {
    unsigned const int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const double cx = centerX + (x - width / 2.0) * scale / width;
    const double cy = centerY + (y - height / 2.0) * scale / width;

    // check if point is in main body to avoid going to MAX_ITER eevry time
    if (const double q = (cx - 0.25) * (cx - 0.25) + cy * cy; q * (q + (cx - 0.25)) <= 0.25 * cy * cy) {
        unsigned const int idx = (y * width + x) * 4;
        color_pixel(maxIter, maxIter, 0.0, 0.0, &pixels[idx], theme);
        return;
    }

    // check second bulb
    if ((cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625) {
        unsigned const int idx = (y * width + x) * 4;
        color_pixel(maxIter, maxIter, 0.0, 0.0, &pixels[idx], theme);
        return;
    }

    double zx = 0.0, zy = 0.0;
    int iter = 0;

    // check periodicity
    double old_zx = 0.0, old_zy = 0.0;
    int period = 0;

    while (zx * zx + zy * zy <= 256.0 && iter < maxIter) {
        const double tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        iter++;

        // check periodicity
        if (fabs(zx - old_zx) < 1e-10 && fabs(zy - old_zy) < 1e-10) {
            iter = maxIter;
            break;
        }

        period++;
        if (constexpr int checkPeriod = 20; period == checkPeriod) {
            period = 0;
            old_zx = zx;
            old_zy = zy;
        }
    }

    unsigned const int idx = (y * width + x) * 4;
    color_pixel(iter, maxIter, zx, zy, &pixels[idx], theme);
}

/**
 * @brief Host wrapper to launch the Mandelbrot CUDA kernel.
 *
 * Configures the grid and block dimensions and launches the kernel.
 *
 * @param pixels_d Device pointer to the pixel array.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param centerX X coordinate of the center of the view.
 * @param centerY Y coordinate of the center of the view.
 * @param scale Scale of the view.
 * @param maxIter Maximum number of iterations.
 * @param theme The color theme to use.
 */
void mandelbrot_cuda(
    uint8_t *pixels_d,
    const int width,
    const int height,
    const double centerX,
    const double centerY,
    const double scale,
    const int maxIter,
    const int theme
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    mandelbrot_kernel<<<grid, block>>>(pixels_d, width, height, centerX, centerY, scale, maxIter, theme);

    cudaDeviceSynchronize();
}
