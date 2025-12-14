#ifndef FRACTAL_CONFIG_H
#define FRACTAL_CONFIG_H

enum class FractalType { MANDELBROT, JULIA, BURNING_SHIP };

struct Config {
    int baseIter = 500;
    FractalType fractalType = FractalType::MANDELBROT;
};

#endif // FRACTAL_CONFIG_H
