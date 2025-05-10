#ifndef NNUE_H
#define NNUE_H

#include <array>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <Eigen/Dense>

void init_session(const char* model_path);
std::array<float, 768> encode_board(const uint64_t* bitboards);
int game_phase(const uint64_t* bitboards);
int evaluate_position(const uint64_t* bitboards);

void load_all_weights();
Eigen::VectorXf encode_board_eigen(const uint64_t* bitboards);
Eigen::VectorXf relu(const Eigen::VectorXf& x);
Eigen::VectorXf load_vector(const std::string& path, int size);
Eigen::MatrixXf load_matrix(const std::string& path, int rows, int cols);
int run_inference(const uint64_t* bitboards);


void load_model();
int run_inference_quantized(const uint64_t* bitboards);
std::array<int16_t, 771> encode_board_quant(const uint64_t* bitboards);

#endif // NNUE_H