#pragma once

__declspec(dllexport) void pairwise_gpu(Eigen::ArrayXXf *distances, Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y, int threads_per_block);