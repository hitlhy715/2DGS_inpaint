#include <torch/extension.h>
#include <tuple>

std::tuple<
    torch::Tensor, // xy
    torch::Tensor> // conics
project_gs_inpaint_bigger_patch_forward(
    torch::Tensor &mean,
    torch::Tensor &L_covariance,
    const int gs_patch,
    const std::tuple<int, int> patch_size,
    const std::tuple<int, int> patch_nums
);


std::tuple<
    torch::Tensor,
    torch::Tensor
>
project_gs_inpaint_bigger_patch_backward(
    const int gsbank_num,
    const std::tuple<int, int> patch_size,
    const std::tuple<int, int> patch_nums,
    torch::Tensor &means,
    torch::Tensor &L_covariance,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
);


torch::Tensor rasterize_gs_inpaint_bigger_patch_forward(
    const int gsbank_num,
    const std::tuple<int, int> patch_size,
    const std::tuple<int, int> img_size,
    const std::tuple<int, int> patch_num,
    torch::Tensor &xys,
    torch::Tensor &conics,
    torch::Tensor &colors,
    torch::Tensor &coefficient
);

std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dconic
    torch::Tensor, // dL_dcolors
    torch::Tensor  // dL_dcoefficient
>
rasterize_gs_inpaint_bigger_patch_backward(
    const std::tuple<int, int> img_size,
    const std::tuple<int, int> patch_size,
    const std::tuple<int, int> patch_nums,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &coefficient,
    const torch::Tensor &v_output // dL_dout_color
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions gaussian inpainting
    m.def("project_gs_inpaint_bigger_patch_forward", &project_gs_inpaint_bigger_patch_forward);
    m.def("project_gs_inpaint_bigger_patch_backward", &project_gs_inpaint_bigger_patch_backward);
    m.def("rasterize_gs_inpaint_bigger_patch_forward", &rasterize_gs_inpaint_bigger_patch_forward);
    m.def("rasterize_gs_inpaint_bigger_patch_backward", &rasterize_gs_inpaint_bigger_patch_backward);
}