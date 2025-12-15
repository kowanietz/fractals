#include "args_parser.h"
#include <iostream>
#include <cstring>

Config parseArgs(const int argc, char *argv[]) {
    Config config;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--iterations") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            config.baseIter = std::atoi(argv[++i]);
            if (config.baseIter <= 0) {
                std::cerr << "Invalid iteration count. Using default (500).\n";
                config.baseIter = 500;
            }
        } else if ((strcmp(argv[i], "--fractal") == 0 || strcmp(argv[i], "-f") == 0) && i + 1 < argc) {
            if (const char *type = argv[++i]; strcmp(type, "julia") == 0) {
                config.fractalType = FractalType::JULIA;
            } else if (strcmp(type, "mandelbrot") == 0) {
                config.fractalType = FractalType::MANDELBROT;
            } else if (strcmp(type, "burningship") == 0) {
                config.fractalType = FractalType::BURNING_SHIP;
            } else {
                std::cerr << "Unknown fractal type. Using mandelbrot.\n";
            }
        } else if ((strcmp(argv[i], "--theme") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            if (const char *theme = argv[++i]; strcmp(theme, "blue_gold") == 0) {
                config.colorTheme = ColorTheme::BLUE_GOLD;
            } else if (strcmp(theme, "rainbow") == 0) {
                config.colorTheme = ColorTheme::RAINBOW;
            } else if (strcmp(theme, "neon") == 0) {
                config.colorTheme = ColorTheme::NEON;
            } else if (strcmp(theme, "fire") == 0) {
                config.colorTheme = ColorTheme::FIRE;
            } else if (strcmp(theme, "ocean") == 0) {
                config.colorTheme = ColorTheme::OCEAN;
            } else if (strcmp(theme, "purple") == 0) {
                config.colorTheme = ColorTheme::PURPLE;
            } else if (strcmp(theme, "grayscale") == 0) {
                config.colorTheme = ColorTheme::GRAYSCALE;
            } else if (strcmp(theme, "electric") == 0) {
                config.colorTheme = ColorTheme::ELECTRIC;
            } else {
                config.colorTheme = ColorTheme::DEFAULT;
            }
        }
    }
    return config;
}
