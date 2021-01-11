#pragma once

__declspec(dllexport) void pairwise_gpu(Eigen::ArrayXf *force_x, Eigen::ArrayXf *force_y, Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y, float SD_factor, int threads_per_block);