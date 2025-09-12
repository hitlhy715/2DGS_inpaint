// by: li hongyu
// description: implement larger patch size

#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <algorithm>
#include <cuda.h>
#include <torch/extension.h>
#include <tuple>
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"

namespace cg = cooperative_groups;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define N_THREADS 256
#define MAX_GS_NUM 1024
#define MAX_PATCH_X 24
#define MAX_PATCH_Y 24

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det < 1.f/(255.f*2))
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    
    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}



__global__ void project_gs_inpaint_bigger_patch_forward_tensor(
    const float2* __restrict__ mean,
    const float3* __restrict__ L_elements,
    const int total_gs_patch,  
    const int current_batch_size, 
    const dim3 patch_num,
    const dim3 patch_size,
    float2* __restrict__ xys,
    float3* __restrict__ conics,
    const int start_idx // add start idx
){
    auto block = cg::this_thread_block();
    unsigned gaussian_idx = block.thread_index().x;
    unsigned patch_idx = block.group_index().x + patch_num.x * block.group_index().y;

    unsigned global_gaussian_idx = start_idx + gaussian_idx;
    unsigned idx = patch_idx*total_gs_patch + global_gaussian_idx; // add start idx

    if (global_gaussian_idx >= total_gs_patch) return; 

    
    float2 center = {0.5f * patch_size.x * mean[idx].x + 0.5f * patch_size.x,
                     0.5f * patch_size.y * mean[idx].y + 0.5f * patch_size.y};


    // inverse matrix
    //  | l11 l12 |
    //  | l21 l22 |
    float l11 = L_elements[idx].x; // scale_x
    float l21 = L_elements[idx].y; // covariance_xy
    float l22 = L_elements[idx].z; // scale_y
    float3 cov2d = make_float3(l11*l11, l11*l21, l21*l21 + l22*l22);

    float3 conic; 
    float radius; 
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    
    xys[idx] = center;
    conics[idx] = conic;

}

std::tuple<
    torch::Tensor, // xy
    torch::Tensor> // conics
project_gs_inpaint_bigger_patch_forward(
    torch::Tensor &mean,
    torch::Tensor &L_covariance,
    const int gs_patch,
    const std::tuple<int, int> patch_size,
    const std::tuple<int, int> patch_nums
){
    dim3 patch_size_dim3;  // e.g 16*16
    patch_size_dim3.x = std::get<0>(patch_size);
    patch_size_dim3.y = std::get<1>(patch_size);

    dim3 patch_num_dim3;  // e.g 16*16
    patch_num_dim3.x = std::get<0>(patch_nums);
    patch_num_dim3.y = std::get<1>(patch_nums);

    int patch_num = patch_num_dim3.x * patch_num_dim3.y;


    torch::Tensor xy_return =
        torch::zeros({patch_num*gs_patch, 2}, mean.options().dtype(torch::kFloat32));
    torch::Tensor conics_return =
        torch::zeros({patch_num*gs_patch, 3}, mean.options().dtype(torch::kFloat32));


    int batch_num = (gs_patch + MAX_GS_NUM - 1) / MAX_GS_NUM;

    for(int batch=0; batch<batch_num; batch++){
        int start_idx = batch*MAX_GS_NUM;
        int current_batch_size = std::min(MAX_GS_NUM, gs_patch-start_idx);

        project_gs_inpaint_bigger_patch_forward_tensor<<<patch_num_dim3, current_batch_size>>>( //modify
            (float2 *) mean.contiguous().data_ptr<float>(),
            (float3 *) L_covariance.contiguous().data_ptr<float>(),
            gs_patch, 
            current_batch_size,
            patch_num_dim3,
            patch_size_dim3,
            (float2 *) xy_return.contiguous().data_ptr<float>(),
            (float3 *) conics_return.contiguous().data_ptr<float>(),
            start_idx
        );
    }

    // wait for GPU
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
    }

    return std::make_tuple(
        xy_return, conics_return
    );

}

__global__ void project_gs_inpaint_bigger_patch_backward_kernel(
    const int gsbank_num,
    const int current_batch_size,
    const dim3 patch_size,
    const dim3 patch_num,
    const float2* __restrict__ means,
    const float3* __restrict__ L_elements,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float3* __restrict__ v_conic,
    float2* __restrict__ v_means,
    float3* __restrict__ v_L_elements,
    const int start_idx 
){
    auto block = cg::this_thread_block();

    unsigned patch_idx = block.group_index().x + patch_num.x * block.group_index().y;
    unsigned gaussian_idx = block.thread_index().x;

    unsigned global_gaussian_idx = start_idx + gaussian_idx;
    unsigned idx = patch_idx*gsbank_num + global_gaussian_idx; // add start idx

    if (global_gaussian_idx >= gsbank_num) return; 


    // means 
    // forward：center.x = (1/2)*patch.x*means.x + (1/2)*patch.x
    // backward: (dL/d means) = (dL/d center.x)*(d center.x/d means.x)
    //                           = v_xy.x * (1/2)* patch.x
    v_means[idx].x = v_xy[idx].x * 0.5f * patch_size.x;
    v_means[idx].y = v_xy[idx].y * 0.5f * patch_size.y;

    float3 cov2d;
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], cov2d);

    // L_elements
    float G_11 = cov2d.x; // dL/dSigma_11
    float G_12 = cov2d.y; // dL/dSigma_12, which is the same as dL/dSigma_21
    float G_22 = cov2d.z; // dL/dSigma_22

    // Extract the individual elements of the L matrix
    float l_11 = L_elements[idx].x; // L_11
    float l_21 = L_elements[idx].y; // L_21
    float l_22 = L_elements[idx].z; // L_22

    // Calculate the gradients with respect to the elements of L
    float grad_l_11 = 2 * l_11 * G_11 + 2 * G_12 * l_21; // dL/dl_11
    float grad_l_21 = 2 * l_11 * G_12 + 2 * l_21 * G_22; // dL/dl_21
    float grad_l_22 = 2 * l_22 * G_22; // dL/dl_22

    // Store the gradients back to the output gradient array
    v_L_elements[idx].x = grad_l_11;
    v_L_elements[idx].y = grad_l_21;
    v_L_elements[idx].z = grad_l_22;

}

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
){

    dim3 patch_size_dim3;
    patch_size_dim3.x = std::get<0>(patch_size);
    patch_size_dim3.y = std::get<1>(patch_size);

    dim3 patch_num_dim3;  
    patch_num_dim3.x = std::get<0>(patch_nums);
    patch_num_dim3.y = std::get<1>(patch_nums);

    int patch_num = patch_num_dim3.x * patch_num_dim3.y;

    torch::Tensor v_means =
        torch::zeros({gsbank_num*patch_num, 2}, means.options().dtype(torch::kFloat32));
    torch::Tensor v_L_elements =
        torch::zeros({gsbank_num*patch_num, 3}, means.options().dtype(torch::kFloat32));

    int batch_num = (gsbank_num + MAX_GS_NUM - 1) / MAX_GS_NUM;

    for(int batch=0; batch<batch_num; batch++){
        int start_idx = batch*MAX_GS_NUM;
        int current_batch_size = std::min(MAX_GS_NUM, gsbank_num-start_idx);

        project_gs_inpaint_bigger_patch_backward_kernel <<<patch_num_dim3, current_batch_size>>>(
            gsbank_num,
            current_batch_size,
            patch_size_dim3,
            patch_num_dim3,
            (float2 *)means.contiguous().data_ptr<float>(),
            (float3 *)L_covariance.contiguous().data_ptr<float>(),
            (float3 *)conics.contiguous().data_ptr<float>(),
            (float2 *)v_xy.contiguous().data_ptr<float>(),
            (float3 *)v_conic.contiguous().data_ptr<float>(),
            // Outputs.
            (float2 *)v_means.contiguous().data_ptr<float>(),
            (float3 *)v_L_elements.contiguous().data_ptr<float>(),
            start_idx
        );

    }
    
    return std::make_tuple(v_means, v_L_elements);
}


__global__ void rasterize_gs_inpaint_forward_bigger_patch_kernel(
    const int gsbank_num,
    const dim3 patch_num_dim3,
    const dim3 patch_size_dim3,
    const dim3 input_block_size3,
    const dim3 img_size_dim3,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ coef,
    float3* __restrict__ out_img
){
    auto block = cg::this_thread_block();

    int32_t patch_id =
        block.group_index().y * patch_num_dim3.x + block.group_index().x;

    int2 gs_range = {
        patch_id * gsbank_num,
        (patch_id + 1) * gsbank_num -1
    };

    unsigned px_base = block.thread_index().x;
    unsigned py_base = block.thread_index().y;
    

    for(unsigned px = px_base; px < patch_size_dim3.x; px+=input_block_size3.x){
        for(unsigned py = py_base; py< patch_size_dim3.y; py+=input_block_size3.y){
            
            int tr = py * blockDim.x + px;  

            // pixel position
            unsigned i = block.group_index().x * patch_size_dim3.x + px;
            unsigned j = block.group_index().y * patch_size_dim3.y + py;

            float3 pixel = {0.f, 0.f, 0.f};

            int32_t pix_id = j*img_size_dim3.x + i;
            
            bool inside = (i < img_size_dim3.x && j < img_size_dim3.y);
            
            for(int t=0; t<gsbank_num; t++){
                int gs_idx = gs_range.x + t;
        
                const float3 conic = conics[gs_idx];
                const float2 xy = xys[gs_idx];
                const float coe = coef[gs_idx];
                const float3 c = colors[gs_idx];
        
                const float2 delta = {xy.x - px, xy.y - py}; // (x,y)-μ
                const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
        
                const float alpha = min(1.f, coe * __expf(-sigma));
        
                pixel.x = pixel.x + c.x*alpha;
                pixel.y = pixel.y + c.y*alpha;
                pixel.z = pixel.z + c.z*alpha;
            }
            if(inside){
                out_img[pix_id] = pixel;
            }
        }
    }

}



torch::Tensor rasterize_gs_inpaint_bigger_patch_forward(
    const int gsbank_num,
    const std::tuple<int, int> patch_size,
    const std::tuple<int, int> img_size,
    const std::tuple<int, int> patch_num,
    torch::Tensor &xys,
    torch::Tensor &conics,
    torch::Tensor &colors,
    torch::Tensor &coefficient
){

    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(coefficient);

    dim3 patch_num_dim3;
    patch_num_dim3.x = std::get<0>(patch_num);
    patch_num_dim3.y = std::get<1>(patch_num);

    dim3 patch_size_dim3;
    patch_size_dim3.x = std::get<0>(patch_size);
    patch_size_dim3.y = std::get<1>(patch_size);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);

    const int channels = colors.size(-1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );

    dim3 input_block_size3 = patch_size_dim3;
    if(patch_size_dim3.x>MAX_PATCH_X){
        input_block_size3.x = MAX_PATCH_X;
    }

    if(patch_size_dim3.y>MAX_PATCH_Y){
        input_block_size3.y = MAX_PATCH_Y;
    }

    rasterize_gs_inpaint_forward_bigger_patch_kernel<<<patch_num_dim3, input_block_size3>>>(
        gsbank_num,
        patch_num_dim3,
        patch_size_dim3,
        input_block_size3,
        img_size_dim3,
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        coefficient.contiguous().data_ptr<float>(),
        (float3 *)out_img.contiguous().data_ptr<float>()
    );

    // wait for GPU
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
    }

    return out_img;
}


__global__ void rasterize_gs_inpaint_bigger_patch_backward_kernel(
    const int gsbank_num,
    const dim3 patch_num,
    const dim3 patch_size,
    const dim3 input_block_size3,
    const dim3 img_size,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ coefficients,
    const float3* __restrict__ v_output,
    // output
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_colors,
    float* __restrict__ v_coefficient
){
    auto block = cg::this_thread_block();

    int32_t patch_id =
        block.group_index().y * patch_num.x + block.group_index().x;

    int2 gs_range = {
        patch_id * gsbank_num,
        (patch_id + 1) * gsbank_num -1
    };


    unsigned px_base = block.thread_index().x;
    unsigned py_base = block.thread_index().y;


    for(unsigned px = px_base; px < patch_size.x; px+=input_block_size3.x){
        for(unsigned py = py_base; py< patch_size.y; py+=input_block_size3.y){
            int tr = py * blockDim.x + px;  
            
            // pixel position
            unsigned i = block.group_index().x * patch_size.x + px;
            unsigned j = block.group_index().y * patch_size.y + py;

            const bool inside = (i < img_size.y && j < img_size.x);

            int32_t pix_id = j*img_size.x + i;
            const float3 v_out = v_output[pix_id];

            for(int t=0; t<gsbank_num; t++){
                int gs_idx = gs_range.x + t;
        
                const float3 conic = conics[gs_idx];
                const float2 xy = xys[gs_idx];
                const float coef = coefficients[gs_idx];
                const float3 rgb = colors[gs_idx];
        
                float2 delta = {xy.x - px, xy.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                float vis = __expf(-sigma);
                float alpha = min(0.99f, coef * vis);

                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    continue;
                }
        
                float3 v_color_local = {0.f, 0.f, 0.f};
                float3 v_conic_local = {0.f, 0.f, 0.f};
                float2 v_xy_local = {0.f, 0.f};
                float v_coefficient_local = 0.f;
        
                const float fac = alpha;
                float v_alpha = 0.f;
                v_color_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};
        
                // contribution from this pixel
                v_alpha += rgb.x * v_out.x;
                v_alpha += rgb.y * v_out.y;
                v_alpha += rgb.z * v_out.z;
        
                const float v_sigma = -coef * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                        0.5f * v_sigma * delta.x * delta.y, 
                                        0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_coefficient_local = vis * v_alpha;
        
                float* v_color_ptr = (float*)(v_colors);

                atomicAdd(v_color_ptr + 3*gs_idx + 0, v_color_local.x);
                atomicAdd(v_color_ptr + 3*gs_idx + 1, v_color_local.y);
                atomicAdd(v_color_ptr + 3*gs_idx + 2, v_color_local.z);
        
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*gs_idx + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*gs_idx + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*gs_idx + 2, v_conic_local.z);
        
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*gs_idx + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*gs_idx + 1, v_xy_local.y);
        
                float* v_coefficient_ptr = (float*)(v_coefficient);
                atomicAdd(v_coefficient_ptr + gs_idx, v_coefficient_local);
        
                block.sync();
            }
        }
    }

}


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
){
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    dim3 patch_num_dim3;  
    patch_num_dim3.x = std::get<0>(patch_nums);
    patch_num_dim3.y = std::get<1>(patch_nums);

    int patch_num = patch_num_dim3.x * patch_num_dim3.y;

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);

    dim3 patch_size_dim3;
    patch_size_dim3.x = std::get<0>(patch_size);
    patch_size_dim3.y = std::get<1>(patch_size);
    
    const int gsbank_num = colors.size(1);
    
    
    torch::Tensor v_xy = torch::zeros({patch_num*gsbank_num, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({patch_num*gsbank_num, 3}, xys.options());
    torch::Tensor v_colors = torch::zeros({patch_num*gsbank_num, 3}, xys.options());
    torch::Tensor v_coefficient = torch::zeros({patch_num*gsbank_num, 1}, xys.options());

    dim3 input_block_size3;
    if(patch_size_dim3.x>MAX_PATCH_X){
        input_block_size3.x = MAX_PATCH_X;
    }else{
        input_block_size3.x = patch_size_dim3.x;
    }

    if(patch_size_dim3.y>MAX_PATCH_Y){
        input_block_size3.y = MAX_PATCH_Y;
    }else{
        input_block_size3.y = patch_size_dim3.y;
    }


    rasterize_gs_inpaint_bigger_patch_backward_kernel<<<patch_num_dim3, input_block_size3>>>(
        gsbank_num,
        patch_num_dim3,
        patch_size_dim3,
        input_block_size3,
        img_size_dim3,
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        (float *)coefficient.contiguous().data_ptr<float>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        // output
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        (float *)v_coefficient.contiguous().data_ptr<float>()
    );


    return std::make_tuple(v_xy, v_conic, v_colors, v_coefficient);
}