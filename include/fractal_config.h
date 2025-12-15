#ifndef FRACTAL_CONFIG_H
#define FRACTAL_CONFIG_H

enum class FractalType { MANDELBROT, JULIA, BURNING_SHIP };

enum class ColorTheme { DEFAULT, BLUE_GOLD, RAINBOW, NEON, FIRE, OCEAN, PURPLE, GRAYSCALE, ELECTRIC };

struct Config {
    int baseIter = 500;
    FractalType fractalType = FractalType::MANDELBROT;
    ColorTheme colorTheme = ColorTheme::DEFAULT;
};

#endif // FRACTAL_CONFIG_H
