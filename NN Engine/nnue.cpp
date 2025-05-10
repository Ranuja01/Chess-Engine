#include <onnxruntime_cxx_api.h>
#include <vector>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <cstddef>  // for size_t, if used
#include <iostream> // for printing, if needed

using namespace Eigen;
using namespace std;

// Enum for piece types
enum PieceType {
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING
};

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NNUE");
Ort::SessionOptions session_options;
std::unique_ptr<Ort::Session> session_ptr;


MatrixXf w1, w2, w3, w4;
VectorXf b1, b2, b3, b4;

using InputVec   = Matrix<float, 771, 1>;
using Layer1Vec  = Matrix<float, 512, 1>;
using Layer2Vec  = Matrix<float, 256, 1>;
using Layer3Vec  = Matrix<float, 128, 1>;
using OutputVec  = Matrix<float, 1, 1>;

using Weights1 = Matrix<float, 512, 771>;
using Weights2 = Matrix<float, 256, 512>;
using Weights3 = Matrix<float, 128, 256>;
using Weights4 = Matrix<float, 1, 128>;


// Variables for integer weights
// Type aliases for convenience

constexpr int INPUT_SIZE = 771;
constexpr int H1_SIZE = 512;
constexpr int H2_SIZE = 256;
constexpr int H3_SIZE = 128;
constexpr int OUT_SIZE = 1;

using Layer1Weights = std::array<std::array<int8_t, INPUT_SIZE>, H1_SIZE>;
using Layer2Weights = std::array<std::array<int8_t, H1_SIZE>, H2_SIZE>;
using Layer3Weights = std::array<std::array<int8_t, H2_SIZE>, H3_SIZE>;
using Layer4Weights = std::array<std::array<int8_t, H3_SIZE>, OUT_SIZE>;

using Bias1 = std::array<int32_t, H1_SIZE>;
using Bias2 = std::array<int32_t, H2_SIZE>;
using Bias3 = std::array<int32_t, H3_SIZE>;
using Bias4 = std::array<int32_t, OUT_SIZE>;

using ActInput = std::array<int16_t, INPUT_SIZE>;
using Act1 = std::array<int16_t, H1_SIZE>;
using Act2 = std::array<int16_t, H2_SIZE>;
using Act3 = std::array<int16_t, H3_SIZE>;

using Accum1 = std::array<int32_t, H1_SIZE>;
using Accum2 = std::array<int32_t, H2_SIZE>;
using Accum3 = std::array<int32_t, H3_SIZE>;
using Accum4 = std::array<int32_t, OUT_SIZE>;

float scale_factors[4]; // To be filled from scales.bin

Layer1Weights w1_quant;
Layer2Weights w2_quant;
Layer3Weights w3_quant;
Layer4Weights w4_quant;

Bias1 b1_quant;
Bias2 b2_quant;
Bias3 b3_quant;
Bias4 b4_quant;

// ReLU activation
template <size_t N>
void relu_quant(const std::array<int32_t, N>& in, std::array<int16_t, N>& out) {
    for (size_t i = 0; i < N; ++i)
        out[i] = std::max<int32_t>(0, in[i]);
}

// Dense layer operation: int8 weights × int16 input → int32 accum
template <size_t OUT, size_t IN>
void dense_layer(const std::array<std::array<int8_t, IN>, OUT>& weights,
                 const std::array<int32_t, OUT>& biases,
                 const std::array<int16_t, IN>& input,
                 std::array<int32_t, OUT>& output) {
    for (size_t i = 0; i < OUT; ++i) {
        int32_t sum = biases[i];
        for (size_t j = 0; j < IN; ++j)
            sum += static_cast<int32_t>(weights[i][j]) * input[j];
        output[i] = sum;
    }
}

// Helper to read binary into a flat array
template <typename T>
void read_binary(const std::string& path, T* buffer, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open " + path);
    file.read(reinterpret_cast<char*>(buffer), count * sizeof(T));
    if (!file) throw std::runtime_error("Failed to read " + path);
}

// Load all model weights
void load_model() {

    const std::string dir = "weights_quant/";

    read_binary(dir + "w1.bin", &w1_quant[0][0], H1_SIZE * INPUT_SIZE);
    read_binary(dir + "b1.bin", &b1_quant[0], H1_SIZE);

    read_binary(dir + "w2.bin", &w2_quant[0][0], H2_SIZE * H1_SIZE);
    read_binary(dir + "b2.bin", &b2_quant[0], H2_SIZE);

    read_binary(dir + "w3.bin", &w3_quant[0][0], H3_SIZE * H2_SIZE);
    read_binary(dir + "b3.bin", &b3_quant[0], H3_SIZE);

    read_binary(dir + "w4.bin", &w4_quant[0][0], OUT_SIZE * H3_SIZE);
    read_binary(dir + "b4.bin", &b4_quant[0], OUT_SIZE);

    read_binary(dir + "/scales.bin", &scale_factors[0], 4);
}

// Load a 1D vector from binary file
Eigen::VectorXf load_vector(const std::string& path, int size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open " + path);

    VectorXf vec(size);
    file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    if (!file) throw std::runtime_error("Failed to read vector from " + path);
    return vec;
}

// Load a 2D matrix from binary file (row-major)
Eigen::MatrixXf load_matrix(const std::string& path, int rows, int cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open " + path);

    MatrixXf mat(rows, cols);
    file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(float));
    if (!file) throw std::runtime_error("Failed to read matrix from " + path);
    return mat;
}

void load_all_weights() {
    const std::string base_dir = "weights/";

    w1 = load_matrix(base_dir + "w1.bin", 512, 771);
    b1 = load_vector(base_dir + "b1.bin", 512);

    w2 = load_matrix(base_dir + "w2.bin", 256, 512);
    b2 = load_vector(base_dir + "b2.bin", 256);

    w3 = load_matrix(base_dir + "w3.bin", 128, 256);
    b3 = load_vector(base_dir + "b3.bin", 128);

    w4 = load_matrix(base_dir + "w4.bin", 1, 128);
    b4 = load_vector(base_dir + "b4.bin", 1);
}

void init_session(const char* model_path) {
    session_options.SetIntraOpNumThreads(12);  // Adjust as needed
    session_ptr = std::make_unique<Ort::Session>(env, model_path, session_options);
}

// Function to calculate the game phase
int game_phase(const uint64_t* bitboards) {
    // Calculate the number of pieces (excluding kings)
    uint64_t combined = bitboards[0];
    for (int i = 1; i < 12; ++i) {
        combined &= bitboards[i];  // AND all bitboards together
    }
    int piece_num = __builtin_popcountll(combined) - 2; // Subtract 2 for the kings

    bool is_endgame;
    bool is_near_gameEnd;

    // Determine the game phase based on piece count and queens
    if (bitboards[1] == 0) {  // Check if there are no queens (bitboard 1 corresponds to queens)
        is_endgame = piece_num < 18;
        is_near_gameEnd = piece_num < 12;
    } else {
        is_endgame = piece_num < 16;
        is_near_gameEnd = piece_num < 10;
    }

    if (is_near_gameEnd) {
        return 0;  // Near endgame phase
    } else if (is_endgame) {
        return 1;  // Endgame phase
    } else {
        return 2;  // Midgame phase
    }
}

std::array<int16_t, 771> encode_board_quant(const uint64_t* bitboards) {
    std::array<int16_t, 771> encoding{};  // 768 for board + 3 for game phase
    const int16_t quantized_one = static_cast<int16_t>(std::round(1.0f / scale_factors[0]));

    for (int piece = 0; piece < 12; ++piece) {
        uint64_t bb = bitboards[piece];
        while (bb) {
            int square = __builtin_ctzll(bb); // Count trailing zeros
            int row = square / 8;
            int col = square % 8;
            int index = row * 8 * 12 + col * 12 + piece;
            encoding[index] = quantized_one;
            bb &= bb - 1; // Clear the least significant bit
        }
    }

    // Set one-hot for game phase
    int phase = game_phase(bitboards);  // should return 0, 1, or 2
    encoding[768 + phase] = quantized_one;

    return encoding;
}

// Function to encode the board into a 771-length array
std::array<float, 771> encode_board(const uint64_t* bitboards) {
    std::array<float, 771> encoding{};  // 768 for board + 3 for game phase
    for (int piece = 0; piece < 12; ++piece) {
        uint64_t bb = bitboards[piece];
        while (bb) {
            int square = __builtin_ctzll(bb); // Count trailing zeros
            int row = square / 8;
            int col = square % 8;
            int index = row * 8 * 12 + col * 12 + piece;
            encoding[index] = 1.0f;
            bb &= bb - 1; // Clear the least significant bit
        }
    }

    // Get game phase and set the appropriate one-hot encoding (index 768-770)
    int phase = game_phase(bitboards);
    encoding[768 + phase] = 1.0f;

    return encoding;
}

Eigen::VectorXf encode_board_eigen(const uint64_t* bitboards) {
    Eigen::VectorXf encoding = Eigen::VectorXf::Zero(771);  // zero-initialize

    for (int piece = 0; piece < 12; ++piece) {
        uint64_t bb = bitboards[piece];
        while (bb) {
            int square = __builtin_ctzll(bb);
            int row = square / 8;
            int col = square % 8;
            int index = row * 8 * 12 + col * 12 + piece;
            encoding[index] = 1.0f;
            bb &= bb - 1;
        }
    }

    // Set one-hot game phase at the end
    int phase = game_phase(bitboards);  // Should return 0, 1, or 2
    encoding[768 + phase] = 1.0f;

    return encoding;
}

// ReLU activation
template <typename Vec>
inline void relu_inplace(Vec& x) {
    x = x.cwiseMax(0.0f);
}

int run_inference(const uint64_t* bitboards) {
    const Eigen::VectorXf input = encode_board_eigen(bitboards);

    Layer1Vec x1 = (w1 * input).colwise() + b1;
    relu_inplace(x1);

    Layer2Vec x2 = (w2 * x1).colwise() + b2;
    relu_inplace(x2);

    Layer3Vec x3 = (w3 * x2).colwise() + b3;
    relu_inplace(x3);

    OutputVec out = (w4 * x3).colwise() + b4;
    return static_cast<int>(out(0) * 10000);
}
    

int evaluate_position(const uint64_t* bitboards) {
    std::array<float, 771> input_data = encode_board(bitboards);

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_shape = {1, 771};
    std::vector<float> input_tensor_values(input_data.begin(), input_data.end());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto input_name = session_ptr->GetInputNameAllocated(0, allocator);
    auto output_name = session_ptr->GetOutputNameAllocated(0, allocator);

    const char* input_name_ptr = input_name.get();
    const char* output_name_ptr = output_name.get();

    auto output_tensors = session_ptr->Run(Ort::RunOptions{nullptr},
                                           &input_name_ptr, &input_tensor, 1,
                                           &output_name_ptr, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    return static_cast<int>(output_data[0] * 10000);
}

// Full inference using quantized weights
int run_inference_quantized(const uint64_t* bitboards) {
    
    std::array<int16_t, 771> input = encode_board_quant(bitboards);

    Act1 x1;
    Act2 x2;
    Act3 x3;

    Accum1 tmp1;
    Accum2 tmp2;
    Accum3 tmp3;
    Accum4 tmp4;

    dense_layer(w1_quant, b1_quant, input, tmp1);
    relu_quant(tmp1, x1);

    dense_layer(w2_quant, b2_quant, x1, tmp2);
    relu_quant(tmp2, x2);

    dense_layer(w3_quant, b3_quant, x2, tmp3);
    relu_quant(tmp3, x3);

    dense_layer(w4_quant, b4_quant, x3, tmp4);

    float final_scale = scale_factors[0] * scale_factors[1] * scale_factors[2] * scale_factors[3];
    return static_cast<int>(std::round(tmp4[0] * final_scale * 10000.0f));
}