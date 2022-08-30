

#include <torch/torch.h>

using namespace torch;

#pragma once;

/*
torch::Tensor create_meshgrid(int height,int width,torch::Device device,
    bool normalized_coordinates = true ) {

    auto xs = torch::linspace(0, width - 1, width, device=device);
    auto ys = torch::linspace(0, height - 1, height, device=device);

    if (normalized_coordinates) {
        xs = (xs / (width - 1) - 0.5) * 2;
        ys = (ys / (height - 1) - 0.5) * 2;
    }

    // generate grid by stacking coordinates
    auto base_grid = torch::stack(torch::meshgrid(torch::TensorList({xs, ys})), 1);  // WxHx2
    return base_grid.permute({1, 0, 2}).unsqueeze(0);  // 1xHxWx2
}

torch::Tensor normalize_pixel_coordinates(
    torch::Tensor pixel_coordinates, int height, int width, float eps = 1e-8) {

    // compute normalization factor
    auto hw = torch::stack(
        {
            torch::empty(width, torch::TensorOptions().device(pixel_coordinates.device()).dtype(pixel_coordinates.dtype())),
            torch::empty(height, torch::TensorOptions().device(pixel_coordinates.device()).dtype(pixel_coordinates.dtype()))
        }
    );

    auto factor = torch.tensor(2.0, torch::TensorOptions().device(pixel_coordinates.device()).dtype(pixel_coordinates.dtype())) / (
        hw - 1
    ).clamp(eps)

    return factor * pixel_coordinates - 1;

}

torch::Tensor _get_window_grid_kernel2d(int h, int w, torch::Device device) {
    /*
    Helper function, which generates a kernel to with window coordinates, residual to window center.
    Args:
         h: kernel height.
         : kernel width.
         device: device, on which generate.
    Returns:
        conv_kernel [2x1xhxw]
    */

    auto window_grid2d = create_meshgrid(h, w, device, false);
    auto window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel
}
*/