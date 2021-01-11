#pragma once

extern "C" __declspec(dllexport) void pairwise_gpu(vector<vector<double>> *distances, vector<double> atoms_x, vector<double> atoms_y, int threads_per_block);