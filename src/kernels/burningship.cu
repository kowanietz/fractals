#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

/**
 * @brief Colors a pixel for the Burning Ship fractal.
 *
 * @param iter Number of iterations.
 * @param maxIter Maximum iterations.
 * @param zx Real part of Z.
 * @param zy Imaginary part of Z.
 * @param pixel Pointer to pixel RGBA.
 * @param theme The color theme to use.
 */
__device__ void color_pixel_burningship(
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
 * @brief CUDA kernel to compute the Burning Ship fractal.
 *
 * @param pixels Output pixel array.
 * @param width Image width.
 * @param height Image height.
 * @param centerX Center X coordinate.
 * @param centerY Center Y coordinate.
 * @param scale View scale.
 * @param maxIter Maximum iterations.
 */
__global__ void burningship_kernel(
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

    double zx = 0.0, zy = 0.0;
    int iter = 0;

    // check periodicity
    double old_zx = 0.0, old_zy = 0.0;
    int period = 0;

    while (zx * zx + zy * zy <= 256.0 && iter < maxIter) {
        const double zx_abs = fabs(zx);
        const double zy_abs = fabs(zy);

        const double tmp = zx_abs * zx_abs - zy_abs * zy_abs + cx;
        zy = 2.0 * zx_abs * zy_abs + cy;
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
    color_pixel_burningship(iter, maxIter, zx, zy, &pixels[idx], theme);
}

/**
 * @brief Host wrapper to launch the Burning Ship CUDA kernel.
 *
 * @param pixels_d Device pointer to pixels.
 * @param width Image width.
 * @param height Image height.
 * @param centerX Center X.
 * @param centerY Center Y.
 * @param scale Scale.
 * @param maxIter Max iterations.
 * @param theme The color theme to use.
 */
void burningship_cuda(
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

    burningship_kernel<<<grid, block>>>(pixels_d, width, height, centerX, centerY, scale, maxIter, theme);

    cudaDeviceSynchronize();
}
