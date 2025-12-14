# CUDA-accelerated fractal renderer

Renders fractals such as the Mandelbrot and Julia sets in an interactive window using CUDA for GPU acceleration.


> [!WARNING]
> CPU rendering is not supported yet. A compatible NVIDIA GPU is required.

[Supported GPUs](https://developer.nvidia.com/cuda/gpus)

## Examples

### Mandelbrot Set

![Mandelbrot](static/mandelbrot.png)

![Mandelbrot 2](static/mandelbrot2.png)

### Julia Set

![Julia.png](static/julia.png)

![Julia 2.png](static/julia2.png)

## Usage

```bash
./run.sh

Options:
  -i, --iterations <iterations>     [int]                 (Default 500)
  -f, --fractal <fractal>           [mandelbrot/julia]    (Default mandelbrot)
  
Example:

./run.sh -f julia -i 1000
```
