#include "Tiny.h"
#include "Tiny_weights_0.h"
#include "Tiny_weights_1.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "esp32-hal-psram.h"
#include "arduino.h"
/*  Operator: Gather 
    Name in model: /token_embedding/Gather
    Input: token_embedding_weight [6000, 64]
    Input: input_ids [1, 64]
    Output: _token_embedding_Gather_output_0 [1, 64, 64]
*/
void node_token_embedding_Gather(const float token_embedding_weight[6000][64], const int64_t input_ids[1][64], float _token_embedding_Gather_output_0[1][64][64]) {
    // Gather along axis=0
    // Input shape: [6000, 64], Indices shape: [1, 64], Output shape: [1, 64, 64]
    for (int d0 = 0; d0 < 1; d0++) {
        for (int d1 = 0; d1 < 64; d1++) {
            for (int d2 = 0; d2 < 64; d2++) {
                int64_t index_val = input_ids[d0][d1];
                if (index_val < 0) index_val += 6000;
                if (index_val < 0 || index_val >= 6000) {
                    // Index out of bounds, set to 0
                    _token_embedding_Gather_output_0[d0][d1][d2] = 0.0f;
                } else {
                    _token_embedding_Gather_output_0[d0][d1][d2] = token_embedding_weight[index_val][d2];
                }
            }
        }
    }
}

/*  Operator: Add 
    Name in model: /Add
    Input: _token_embedding_Gather_output_0 [1, 64, 64]
    Input: onnx_Add_140 [64, 64]
    Output: _Add_output_0 [1, 64, 64]
*/
void node_Add(const float _token_embedding_Gather_output_0[1][64][64], const float onnx_Add_140[64][64], float _Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _Add_output_0[d0][d1][d2] = _token_embedding_Gather_output_0[d0][d1][d2] + onnx_Add_140[d1][d2];
    }
    }
    }
}

/*  Operator: LayerNormalization 
    Name in model: /blocks.0/ln1/LayerNormalization
    Input: _Add_output_0 [1, 64, 64]
    Input: blocks_0_ln1_weight [64]
    Input: blocks_0_ln1_bias [64]
    Output: _blocks_0_ln1_LayerNormalization_output_0 [1, 64, 64]
*/
void node_blocks_0_ln1_LayerNormalization(const float _Add_output_0[1][64][64], const float blocks_0_ln1_weight[64], const float blocks_0_ln1_bias[64], float _blocks_0_ln1_LayerNormalization_output_0[1][64][64]) {
    const int D0 = 1;
    const int D1 = 64;
    const int D2 = 64;

    // LayerNormalization: axis=-1, epsilon=9.999999747378752e-06
    const int INNER_SIZE = 64;
    for (int d0 = 0; d0 < D0; ++d0) {
    for (int d1 = 0; d1 < D1; ++d1) {
        /* compute mean */
        float mean = 0.0f;
        for (int i = 0; i < INNER_SIZE; ++i) {
            mean += _Add_output_0[d0][d1][i];
        }
        mean /= (float)INNER_SIZE;

        /* compute variance */
        float var = 0.0f;
        for (int i = 0; i < INNER_SIZE; ++i) {
            float diff = _Add_output_0[d0][d1][i] - mean;
            var += diff * diff;
        }
        var /= (float)INNER_SIZE;
        float inv_std = 1.0f / sqrtf(var + 9.999999747378752e-06f);

        /* normalize, scale and bias */
        for (int i = 0; i < INNER_SIZE; ++i) {
            float normalized = (_Add_output_0[d0][d1][i] - mean) * inv_std;
            _blocks_0_ln1_LayerNormalization_output_0[d0][d1][i] = normalized * blocks_0_ln1_weight[i] + blocks_0_ln1_bias[i];
        }
    }
    }
}

/*  Operator: MatMul 
    Name in model: MatMul_3
    Input: _blocks_0_ln1_LayerNormalization_output_0 [1, 64, 64]
    Input: _v_92 [64, 192]
    Output: _v_93 [1, 64, 192]
*/
void node_MatMul_3(const float _blocks_0_ln1_LayerNormalization_output_0[1][64][64], const float _v_92[64][192], float _v_93[1][64][192]) {
    /*Matrix multiplication*/
    for (int b = 0; b < 1; b++) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 192; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 64; k++) {
                    acc += _blocks_0_ln1_LayerNormalization_output_0[b][i][k] * _v_92[k][j];
                }
            _v_93[b][i][j] = acc;
            }
        }
    }
}

/*  Operator: Split 
    Name in model: Split_4
    Input: _v_93 [1, 64, 192]
    Input: _v_97 [3]
    Output: _v_94 [1, 64, 64]    Output: _v_95 [1, 64, 64]    Output: _v_96 [1, 64, 64]
*/
void node_Split_4(const float _v_93[1][64][192], const int64_t _v_97[3], float _v_94[1][64][64], float _v_95[1][64][64], float _v_96[1][64][64]) {
    // Split along axis=-1 (normalized to 2)
    const int64_t* split = (const int64_t*)_v_97;

    // Processing output 0: _v_94
    const int64_t start_0 = 0;
    const int64_t split_size_0 = split[0];

    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 64; i1++) {
            for (int i2 = 0; i2 < split_size_0; i2++) {
                _v_94[i0][i1][i2] = _v_93[i0][i1][start_0 + i2];
            }
        }
    }

    // Processing output 1: _v_95
    const int64_t start_1 = split[0];
    const int64_t split_size_1 = split[1];

    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 64; i1++) {
            for (int i2 = 0; i2 < split_size_1; i2++) {
                _v_95[i0][i1][i2] = _v_93[i0][i1][start_1 + i2];
            }
        }
    }

    // Processing output 2: _v_96
    const int64_t start_2 = split[0] + split[1];
    const int64_t split_size_2 = split[2];

    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 64; i1++) {
            for (int i2 = 0; i2 < split_size_2; i2++) {
                _v_96[i0][i1][i2] = _v_93[i0][i1][start_2 + i2];
            }
        }
    }

}

/*  Operator: Add 
    Name in model: /blocks.0/attn/key/Add
    Input: blocks_0_attn_key_bias [64]
    Input: _v_94 [1, 64, 64]
    Output: _blocks_0_attn_key_Add_output_0 [1, 64, 64]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_attn_key_Add(const float blocks_0_attn_key_bias[64], const float _v_94[1][64][64], float _blocks_0_attn_key_Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_attn_key_Add_output_0[d0][d1][d2] = blocks_0_attn_key_bias[d2] + _v_94[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Reshape 
    Name in model: /blocks.0/attn/Reshape
    Input: _blocks_0_attn_key_Add_output_0 [1, 64, 64]
    Input: _blocks_0_attn_Constant_output_0 [4]
    Output: _blocks_0_attn_Reshape_output_0 [1, 64, 4, 16]
*/
void node_blocks_0_attn_Reshape(const float _blocks_0_attn_key_Add_output_0[1][64][64], const int64_t _blocks_0_attn_Constant_output_0[4], float _blocks_0_attn_Reshape_output_0[1][64][4][16]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_blocks_0_attn_key_Add_output_0;
    float *dst = (float*)_blocks_0_attn_Reshape_output_0;
    memcpy(dst, src, 4096 * sizeof(float));
}

/*  Operator: Add 
    Name in model: /blocks.0/attn/query/Add
    Input: blocks_0_attn_query_bias [64]
    Input: _v_95 [1, 64, 64]
    Output: _blocks_0_attn_query_Add_output_0 [1, 64, 64]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_attn_query_Add(const float blocks_0_attn_query_bias[64], const float _v_95[1][64][64], float _blocks_0_attn_query_Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_attn_query_Add_output_0[d0][d1][d2] = blocks_0_attn_query_bias[d2] + _v_95[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Reshape 
    Name in model: /blocks.0/attn/Reshape_1
    Input: _blocks_0_attn_query_Add_output_0 [1, 64, 64]
    Input: _blocks_0_attn_Constant_output_0 [4]
    Output: _blocks_0_attn_Reshape_1_output_0 [1, 64, 4, 16]
*/
void node_blocks_0_attn_Reshape_1(const float _blocks_0_attn_query_Add_output_0[1][64][64], const int64_t _blocks_0_attn_Constant_output_0[4], float _blocks_0_attn_Reshape_1_output_0[1][64][4][16]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_blocks_0_attn_query_Add_output_0;
    float *dst = (float*)_blocks_0_attn_Reshape_1_output_0;
    memcpy(dst, src, 4096 * sizeof(float));
}

/*  Operator: Transpose 
    Name in model: /blocks.0/attn/Transpose
    Input: _blocks_0_attn_Reshape_1_output_0 [1, 64, 4, 16]
    Output: _blocks_0_attn_Transpose_output_0 [1, 4, 64, 16]
*/
void node_blocks_0_attn_Transpose(const float _blocks_0_attn_Reshape_1_output_0[1][64][4][16], float _blocks_0_attn_Transpose_output_0[1][4][64][16]) {
    /*Transpose with perm = [0, 2, 1, 3]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i2 = 0; i2 < 64; i2++) {
                for (int i3 = 0; i3 < 16; i3++) {
                    _blocks_0_attn_Transpose_output_0[i0][i1][i2][i3] = _blocks_0_attn_Reshape_1_output_0[i0][i2][i1][i3];
                }
            }
        }
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/attn/value/Add
    Input: blocks_0_attn_value_bias [64]
    Input: _v_96 [1, 64, 64]
    Output: _blocks_0_attn_value_Add_output_0 [1, 64, 64]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_attn_value_Add(const float blocks_0_attn_value_bias[64], const float _v_96[1][64][64], float _blocks_0_attn_value_Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_attn_value_Add_output_0[d0][d1][d2] = blocks_0_attn_value_bias[d2] + _v_96[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Reshape 
    Name in model: /blocks.0/attn/Reshape_2
    Input: _blocks_0_attn_value_Add_output_0 [1, 64, 64]
    Input: _blocks_0_attn_Constant_output_0 [4]
    Output: _blocks_0_attn_Reshape_2_output_0 [1, 64, 4, 16]
*/
void node_blocks_0_attn_Reshape_2(const float _blocks_0_attn_value_Add_output_0[1][64][64], const int64_t _blocks_0_attn_Constant_output_0[4], float _blocks_0_attn_Reshape_2_output_0[1][64][4][16]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_blocks_0_attn_value_Add_output_0;
    float *dst = (float*)_blocks_0_attn_Reshape_2_output_0;
    memcpy(dst, src, 4096 * sizeof(float));
}

/*  Operator: Transpose 
    Name in model: /blocks.0/attn/Transpose_1
    Input: _blocks_0_attn_Reshape_2_output_0 [1, 64, 4, 16]
    Output: _blocks_0_attn_Transpose_1_output_0 [1, 4, 64, 16]
*/
void node_blocks_0_attn_Transpose_1(const float _blocks_0_attn_Reshape_2_output_0[1][64][4][16], float _blocks_0_attn_Transpose_1_output_0[1][4][64][16]) {
    /*Transpose with perm = [0, 2, 1, 3]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i2 = 0; i2 < 64; i2++) {
                for (int i3 = 0; i3 < 16; i3++) {
                    _blocks_0_attn_Transpose_1_output_0[i0][i1][i2][i3] = _blocks_0_attn_Reshape_2_output_0[i0][i2][i1][i3];
                }
            }
        }
    }
}

/*  Operator: Transpose 
    Name in model: /blocks.0/attn/Transpose_2
    Input: _blocks_0_attn_Reshape_output_0 [1, 64, 4, 16]
    Output: _blocks_0_attn_Transpose_2_output_0 [1, 4, 16, 64]
*/
void node_blocks_0_attn_Transpose_2(const float _blocks_0_attn_Reshape_output_0[1][64][4][16], float _blocks_0_attn_Transpose_2_output_0[1][4][16][64]) {
    /*Transpose with perm = [0, 2, 3, 1]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i2 = 0; i2 < 16; i2++) {
                for (int i3 = 0; i3 < 64; i3++) {
                    _blocks_0_attn_Transpose_2_output_0[i0][i1][i2][i3] = _blocks_0_attn_Reshape_output_0[i0][i3][i1][i2];
                }
            }
        }
    }
}

/*  Operator: MatMul 
    Name in model: /blocks.0/attn/MatMul
    Input: _blocks_0_attn_Transpose_output_0 [1, 4, 64, 16]
    Input: _blocks_0_attn_Transpose_2_output_0 [1, 4, 16, 64]
    Output: _blocks_0_attn_MatMul_output_0 [1, 4, 64, 64]
*/
void node_blocks_0_attn_MatMul(const float _blocks_0_attn_Transpose_output_0[1][4][64][16], const float _blocks_0_attn_Transpose_2_output_0[1][4][16][64], float _blocks_0_attn_MatMul_output_0[1][4][64][64]) {
    /*Matrix multiplication*/
    for (int b1 = 0; b1 < 1; b1++) {
        for (int b2 = 0; b2 < 4; b2++) {
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 64; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        acc += _blocks_0_attn_Transpose_output_0[b1][b2][i][k] * _blocks_0_attn_Transpose_2_output_0[b1][b2][k][j];
                    }
                 _blocks_0_attn_MatMul_output_0[b1][b2][i][j] = acc; 
                }
            }
        }
    }
}

/*  Operator: Mul 
    Name in model: /blocks.0/attn/Mul
    Input: _blocks_0_attn_MatMul_output_0 [1, 4, 64, 64]
    Input: _blocks_0_attn_Constant_3_output_0 [1]
    Output: _blocks_0_attn_Mul_output_0 [1, 4, 64, 64]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_attn_Mul(const float _blocks_0_attn_MatMul_output_0[1][4][64][64], const float _blocks_0_attn_Constant_3_output_0[1], float _blocks_0_attn_Mul_output_0[1][4][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 4; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
    for (int d3 = 0; d3 < 64; d3++) {
        _blocks_0_attn_Mul_output_0[d0][d1][d2][d3] = _blocks_0_attn_MatMul_output_0[d0][d1][d2][d3] * _blocks_0_attn_Constant_3_output_0[0];
    }
    }
    }
    }
}

/*  Operator: Where 
    Name in model: /blocks.0/attn/Where
    Input: onnx_Where_170 [1, 1, 64, 64]
    Input: _blocks_0_attn_Constant_4_output_0 [1]
    Input: _blocks_0_attn_Mul_output_0 [1, 4, 64, 64]
    Output: _blocks_0_attn_Where_output_0 [1, 4, 64, 64]
*/
void node_blocks_0_attn_Where(const bool onnx_Where_170[1][1][64][64], const float _blocks_0_attn_Constant_4_output_0[1], const float _blocks_0_attn_Mul_output_0[1][4][64][64], float _blocks_0_attn_Where_output_0[1][4][64][64]) {
    /*where*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i2 = 0; i2 < 64; i2++) {
                for (int i3 = 0; i3 < 64; i3++) {
                    _blocks_0_attn_Where_output_0[i0][i1][i2][i3] = onnx_Where_170[i0][0][i2][i3] ? _blocks_0_attn_Constant_4_output_0[0] : _blocks_0_attn_Mul_output_0[i0][i1][i2][i3];
                }
            }
        }
    }
}

/*  Operator: Softmax 
    Name in model: /blocks.0/attn/Softmax
    Input: _blocks_0_attn_Where_output_0 [1, 4, 64, 64]
    Output: _blocks_0_attn_Softmax_output_0 [1, 4, 64, 64]
*/
void node_blocks_0_attn_Softmax(const float _blocks_0_attn_Where_output_0[1][4][64][64], float _blocks_0_attn_Softmax_output_0[1][4][64][64]) {
    /*Softmax operator implementation along axis 3*/
    const int D0 = 1;
    const int D1 = 4;
    const int D2 = 64;
    const int D3 = 64;

    for (int i0 = 0; i0 < D0; i0++) {
        for (int i1 = 0; i1 < D1; i1++) {
            for (int i2 = 0; i2 < D2; i2++) {
                // Find maximum value for numerical stability
                float max_val = -INFINITY;
                for (int a = 0; a < D3; a++) {
                    if (_blocks_0_attn_Where_output_0[i0][i1][i2][a] > max_val) { max_val = _blocks_0_attn_Where_output_0[i0][i1][i2][a]; }
                }

                // Compute sum of exponentials along the softmax axis
                float sum = 0.0f;
                for (int a = 0; a < D3; a++) {
                    float exp_val = expf(_blocks_0_attn_Where_output_0[i0][i1][i2][a] - max_val);
                    sum += exp_val;
                }

                // Compute softmax by normalizing the exponentials
                for (int a = 0; a < D3; a++) {
                    _blocks_0_attn_Softmax_output_0[i0][i1][i2][a] = expf(_blocks_0_attn_Where_output_0[i0][i1][i2][a] - max_val) / sum;
                }
            }
        }
    }
}

/*  Operator: MatMul 
    Name in model: /blocks.0/attn/MatMul_1
    Input: _blocks_0_attn_Softmax_output_0 [1, 4, 64, 64]
    Input: _blocks_0_attn_Transpose_1_output_0 [1, 4, 64, 16]
    Output: _blocks_0_attn_MatMul_1_output_0 [1, 4, 64, 16]
*/
void node_blocks_0_attn_MatMul_1(const float _blocks_0_attn_Softmax_output_0[1][4][64][64], const float _blocks_0_attn_Transpose_1_output_0[1][4][64][16], float _blocks_0_attn_MatMul_1_output_0[1][4][64][16]) {
    /*Matrix multiplication*/
    for (int b1 = 0; b1 < 1; b1++) {
        for (int b2 = 0; b2 < 4; b2++) {
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 16; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < 64; k++) {
                        acc += _blocks_0_attn_Softmax_output_0[b1][b2][i][k] * _blocks_0_attn_Transpose_1_output_0[b1][b2][k][j];
                    }
                 _blocks_0_attn_MatMul_1_output_0[b1][b2][i][j] = acc; 
                }
            }
        }
    }
}

/*  Operator: Transpose 
    Name in model: /blocks.0/attn/Transpose_3
    Input: _blocks_0_attn_MatMul_1_output_0 [1, 4, 64, 16]
    Output: _blocks_0_attn_Transpose_3_output_0 [1, 64, 4, 16]
*/
void node_blocks_0_attn_Transpose_3(const float _blocks_0_attn_MatMul_1_output_0[1][4][64][16], float _blocks_0_attn_Transpose_3_output_0[1][64][4][16]) {
    /*Transpose with perm = [0, 2, 1, 3]*/
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 64; i1++) {
            for (int i2 = 0; i2 < 4; i2++) {
                for (int i3 = 0; i3 < 16; i3++) {
                    _blocks_0_attn_Transpose_3_output_0[i0][i1][i2][i3] = _blocks_0_attn_MatMul_1_output_0[i0][i2][i1][i3];
                }
            }
        }
    }
}

/*  Operator: Reshape 
    Name in model: /blocks.0/attn/Reshape_3
    Input: _blocks_0_attn_Transpose_3_output_0 [1, 64, 4, 16]
    Input: _blocks_0_attn_Constant_5_output_0 [3]
    Output: _blocks_0_attn_Reshape_3_output_0 [1, 64, 64]
*/
void node_blocks_0_attn_Reshape_3(const float _blocks_0_attn_Transpose_3_output_0[1][64][4][16], const int64_t _blocks_0_attn_Constant_5_output_0[3], float _blocks_0_attn_Reshape_3_output_0[1][64][64]) {
    // Reshape does not modify data, only strides/shapes
    float *src = (float*)_blocks_0_attn_Transpose_3_output_0;
    float *dst = (float*)_blocks_0_attn_Reshape_3_output_0;
    memcpy(dst, src, 4096 * sizeof(float));
}

/*  Operator: MatMul 
    Name in model: /blocks.0/attn/proj/MatMul
    Input: _blocks_0_attn_Reshape_3_output_0 [1, 64, 64]
    Input: onnx_MatMul_175 [64, 64]
    Output: _blocks_0_attn_proj_MatMul_output_0 [1, 64, 64]
*/
void node_blocks_0_attn_proj_MatMul(const float _blocks_0_attn_Reshape_3_output_0[1][64][64], const float onnx_MatMul_175[64][64], float _blocks_0_attn_proj_MatMul_output_0[1][64][64]) {
    /*Matrix multiplication*/
    for (int b = 0; b < 1; b++) {
        for (int i = 0; i <  64; i++) {
            for (int j = 0; j < 64; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 64; k++) {
                    acc += _blocks_0_attn_Reshape_3_output_0[b][i][k] * onnx_MatMul_175[k][j];
                }
            _blocks_0_attn_proj_MatMul_output_0[b][i][j] = acc;
            }
        }
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/attn/proj/Add
    Input: blocks_0_attn_proj_bias [64]
    Input: _blocks_0_attn_proj_MatMul_output_0 [1, 64, 64]
    Output: _blocks_0_attn_proj_Add_output_0 [1, 64, 64]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_attn_proj_Add(const float blocks_0_attn_proj_bias[64], const float _blocks_0_attn_proj_MatMul_output_0[1][64][64], float _blocks_0_attn_proj_Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_attn_proj_Add_output_0[d0][d1][d2] = blocks_0_attn_proj_bias[d2] + _blocks_0_attn_proj_MatMul_output_0[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/Add
    Input: _Add_output_0 [1, 64, 64]
    Input: _blocks_0_attn_proj_Add_output_0 [1, 64, 64]
    Output: _blocks_0_Add_output_0 [1, 64, 64]
*/
void node_blocks_0_Add(const float _Add_output_0[1][64][64], const float _blocks_0_attn_proj_Add_output_0[1][64][64], float _blocks_0_Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_Add_output_0[d0][d1][d2] = _Add_output_0[d0][d1][d2] + _blocks_0_attn_proj_Add_output_0[d0][d1][d2];
    }
    }
    }
}

/*  Operator: LayerNormalization 
    Name in model: /blocks.0/ln2/LayerNormalization
    Input: _blocks_0_Add_output_0 [1, 64, 64]
    Input: blocks_0_ln2_weight [64]
    Input: blocks_0_ln2_bias [64]
    Output: _blocks_0_ln2_LayerNormalization_output_0 [1, 64, 64]
*/
void node_blocks_0_ln2_LayerNormalization(const float _blocks_0_Add_output_0[1][64][64], const float blocks_0_ln2_weight[64], const float blocks_0_ln2_bias[64], float _blocks_0_ln2_LayerNormalization_output_0[1][64][64]) {
    const int D0 = 1;
    const int D1 = 64;
    const int D2 = 64;

    // LayerNormalization: axis=-1, epsilon=9.999999747378752e-06
    const int INNER_SIZE = 64;
    for (int d0 = 0; d0 < D0; ++d0) {
    for (int d1 = 0; d1 < D1; ++d1) {
        /* compute mean */
        float mean = 0.0f;
        for (int i = 0; i < INNER_SIZE; ++i) {
            mean += _blocks_0_Add_output_0[d0][d1][i];
        }
        mean /= (float)INNER_SIZE;

        /* compute variance */
        float var = 0.0f;
        for (int i = 0; i < INNER_SIZE; ++i) {
            float diff = _blocks_0_Add_output_0[d0][d1][i] - mean;
            var += diff * diff;
        }
        var /= (float)INNER_SIZE;
        float inv_std = 1.0f / sqrtf(var + 9.999999747378752e-06f);

        /* normalize, scale and bias */
        for (int i = 0; i < INNER_SIZE; ++i) {
            float normalized = (_blocks_0_Add_output_0[d0][d1][i] - mean) * inv_std;
            _blocks_0_ln2_LayerNormalization_output_0[d0][d1][i] = normalized * blocks_0_ln2_weight[i] + blocks_0_ln2_bias[i];
        }
    }
    }
}

/*  Operator: MatMul 
    Name in model: /blocks.0/ffn/ffn.0/MatMul
    Input: _blocks_0_ln2_LayerNormalization_output_0 [1, 64, 64]
    Input: onnx_MatMul_176 [64, 256]
    Output: _blocks_0_ffn_ffn_0_MatMul_output_0 [1, 64, 256]
*/
void node_blocks_0_ffn_ffn_0_MatMul(const float _blocks_0_ln2_LayerNormalization_output_0[1][64][64], const float onnx_MatMul_176[64][256], float _blocks_0_ffn_ffn_0_MatMul_output_0[1][64][256]) {
    /*Matrix multiplication*/
    for (int b = 0; b < 1; b++) {
        for (int i = 0; i <  64; i++) {
            for (int j = 0; j < 256; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 64; k++) {
                    acc += _blocks_0_ln2_LayerNormalization_output_0[b][i][k] * onnx_MatMul_176[k][j];
                }
            _blocks_0_ffn_ffn_0_MatMul_output_0[b][i][j] = acc;
            }
        }
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/ffn/ffn.0/Add
    Input: blocks_0_ffn_0_bias [256]
    Input: _blocks_0_ffn_ffn_0_MatMul_output_0 [1, 64, 256]
    Output: _blocks_0_ffn_ffn_0_Add_output_0 [1, 64, 256]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_ffn_ffn_0_Add(const float blocks_0_ffn_0_bias[256], const float _blocks_0_ffn_ffn_0_MatMul_output_0[1][64][256], float _blocks_0_ffn_ffn_0_Add_output_0[1][64][256]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 256; d2++) {
        _blocks_0_ffn_ffn_0_Add_output_0[d0][d1][d2] = blocks_0_ffn_0_bias[d2] + _blocks_0_ffn_ffn_0_MatMul_output_0[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Div 
    Name in model: /blocks.0/ffn/ffn.1/Div
    Input: _blocks_0_ffn_ffn_0_Add_output_0 [1, 64, 256]
    Input: _blocks_0_ffn_ffn_1_Constant_output_0 [1]
    Output: _blocks_0_ffn_ffn_1_Div_output_0 [1, 64, 256]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_ffn_ffn_1_Div(const float _blocks_0_ffn_ffn_0_Add_output_0[1][64][256], const float _blocks_0_ffn_ffn_1_Constant_output_0[1], float _blocks_0_ffn_ffn_1_Div_output_0[1][64][256]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 <  64; d1++) {
    for (int d2 = 0; d2 < 256; d2++) {
        _blocks_0_ffn_ffn_1_Div_output_0[d0][d1][d2] = _blocks_0_ffn_ffn_0_Add_output_0[d0][d1][d2] / _blocks_0_ffn_ffn_1_Constant_output_0[0];
    }
    }
    }
}

/*  Operator: Erf 
    Name in model: /blocks.0/ffn/ffn.1/Erf
    Input: _blocks_0_ffn_ffn_1_Div_output_0 [1, 64, 256]
    Output: _blocks_0_ffn_ffn_1_Erf_output_0 [1, 64, 256]
*/
void node_blocks_0_ffn_ffn_1_Erf(const float _blocks_0_ffn_ffn_1_Div_output_0[1][64][256], float _blocks_0_ffn_ffn_1_Erf_output_0[1][64][256]) {
    /* element-wise error function (erf) */
    /* Note: requires <math.h> at top-level for erf/erff */

   float *src = (float*)_blocks_0_ffn_ffn_1_Div_output_0;
   float *dst = (float*)_blocks_0_ffn_ffn_1_Erf_output_0;
    size_t total = 16384u;
    if (total == 0) return;
    for (size_t i = 0; i < total; ++i) {
        dst[i] = erff(src[i]);
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/ffn/ffn.1/Add
    Input: _blocks_0_ffn_ffn_1_Erf_output_0 [1, 64, 256]
    Input: _blocks_0_ffn_ffn_1_Constant_1_output_0 [1]
    Output: _blocks_0_ffn_ffn_1_Add_output_0 [1, 64, 256]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_ffn_ffn_1_Add(const float _blocks_0_ffn_ffn_1_Erf_output_0[1][64][256], const float _blocks_0_ffn_ffn_1_Constant_1_output_0[1], float _blocks_0_ffn_ffn_1_Add_output_0[1][64][256]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 256; d2++) {
        _blocks_0_ffn_ffn_1_Add_output_0[d0][d1][d2] = _blocks_0_ffn_ffn_1_Erf_output_0[d0][d1][d2] + _blocks_0_ffn_ffn_1_Constant_1_output_0[0];
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /blocks.0/ffn/ffn.1/Mul
    Input: _blocks_0_ffn_ffn_0_Add_output_0 [1, 64, 256]
    Input: _blocks_0_ffn_ffn_1_Add_output_0 [1, 64, 256]
    Output: _blocks_0_ffn_ffn_1_Mul_output_0 [1, 64, 256]
*/
void node_blocks_0_ffn_ffn_1_Mul(const float _blocks_0_ffn_ffn_0_Add_output_0[1][64][256], const float _blocks_0_ffn_ffn_1_Add_output_0[1][64][256], float _blocks_0_ffn_ffn_1_Mul_output_0[1][64][256]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 256; d2++) {
        _blocks_0_ffn_ffn_1_Mul_output_0[d0][d1][d2] = _blocks_0_ffn_ffn_0_Add_output_0[d0][d1][d2] * _blocks_0_ffn_ffn_1_Add_output_0[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Mul 
    Name in model: /blocks.0/ffn/ffn.1/Mul_1
    Input: _blocks_0_ffn_ffn_1_Mul_output_0 [1, 64, 256]
    Input: _blocks_0_ffn_ffn_1_Constant_2_output_0 [1]
    Output: _blocks_0_ffn_ffn_1_Mul_1_output_0 [1, 64, 256]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_ffn_ffn_1_Mul_1(const float _blocks_0_ffn_ffn_1_Mul_output_0[1][64][256], const float _blocks_0_ffn_ffn_1_Constant_2_output_0[1], float _blocks_0_ffn_ffn_1_Mul_1_output_0[1][64][256]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 256; d2++) {
        _blocks_0_ffn_ffn_1_Mul_1_output_0[d0][d1][d2] = _blocks_0_ffn_ffn_1_Mul_output_0[d0][d1][d2] * _blocks_0_ffn_ffn_1_Constant_2_output_0[0];
    }
    }
    }
}

/*  Operator: MatMul 
    Name in model: /blocks.0/ffn/ffn.2/MatMul
    Input: _blocks_0_ffn_ffn_1_Mul_1_output_0 [1, 64, 256]
    Input: onnx_MatMul_177 [256, 64]
    Output: _blocks_0_ffn_ffn_2_MatMul_output_0 [1, 64, 64]
*/
void node_blocks_0_ffn_ffn_2_MatMul(const float _blocks_0_ffn_ffn_1_Mul_1_output_0[1][64][256], const float onnx_MatMul_177[256][64], float _blocks_0_ffn_ffn_2_MatMul_output_0[1][64][64]) {
    /*Matrix multiplication*/
    for (int b = 0; b < 1; b++) {
        for (int i = 0; i <  64; i++) {
            for (int j = 0; j < 64; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 256; k++) {
                    acc += _blocks_0_ffn_ffn_1_Mul_1_output_0[b][i][k] * onnx_MatMul_177[k][j];
                }
            _blocks_0_ffn_ffn_2_MatMul_output_0[b][i][j] = acc;
            }
        }
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/ffn/ffn.2/Add
    Input: blocks_0_ffn_2_bias [64]
    Input: _blocks_0_ffn_ffn_2_MatMul_output_0 [1, 64, 64]
    Output: _blocks_0_ffn_ffn_2_Add_output_0 [1, 64, 64]
*/
    // Warning: Broadcasting is applied.
void node_blocks_0_ffn_ffn_2_Add(const float blocks_0_ffn_2_bias[64], const float _blocks_0_ffn_ffn_2_MatMul_output_0[1][64][64], float _blocks_0_ffn_ffn_2_Add_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_ffn_ffn_2_Add_output_0[d0][d1][d2] = blocks_0_ffn_2_bias[d2] + _blocks_0_ffn_ffn_2_MatMul_output_0[d0][d1][d2];
    }
    }
    }
}

/*  Operator: Add 
    Name in model: /blocks.0/Add_1
    Input: _blocks_0_Add_output_0 [1, 64, 64]
    Input: _blocks_0_ffn_ffn_2_Add_output_0 [1, 64, 64]
    Output: _blocks_0_Add_1_output_0 [1, 64, 64]
*/
void node_blocks_0_Add_1(const float _blocks_0_Add_output_0[1][64][64], const float _blocks_0_ffn_ffn_2_Add_output_0[1][64][64], float _blocks_0_Add_1_output_0[1][64][64]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 64; d2++) {
        _blocks_0_Add_1_output_0[d0][d1][d2] = _blocks_0_Add_output_0[d0][d1][d2] + _blocks_0_ffn_ffn_2_Add_output_0[d0][d1][d2];
    }
    }
    }
}

/*  Operator: LayerNormalization 
    Name in model: /ln_f/LayerNormalization
    Input: _blocks_0_Add_1_output_0 [1, 64, 64]
    Input: ln_f_weight [64]
    Input: ln_f_bias [64]
    Output: _ln_f_LayerNormalization_output_0 [1, 64, 64]
*/
void node_ln_f_LayerNormalization(const float _blocks_0_Add_1_output_0[1][64][64], const float ln_f_weight[64], const float ln_f_bias[64], float _ln_f_LayerNormalization_output_0[1][64][64]) {
    const int D0 = 1;
    const int D1 = 64;
    const int D2 = 64;

    // LayerNormalization: axis=-1, epsilon=9.999999747378752e-06
    const int INNER_SIZE = 64;
    for (int d0 = 0; d0 < D0; ++d0) {
    for (int d1 = 0; d1 < D1; ++d1) {
        /* compute mean */
        float mean = 0.0f;
        for (int i = 0; i < INNER_SIZE; ++i) {
            mean += _blocks_0_Add_1_output_0[d0][d1][i];
        }
        mean /= (float)INNER_SIZE;

        /* compute variance */
        float var = 0.0f;
        for (int i = 0; i < INNER_SIZE; ++i) {
            float diff = _blocks_0_Add_1_output_0[d0][d1][i] - mean;
            var += diff * diff;
        }
        var /= (float)INNER_SIZE;
        float inv_std = 1.0f / sqrtf(var + 9.999999747378752e-06f);

        /* normalize, scale and bias */
        for (int i = 0; i < INNER_SIZE; ++i) {
            float normalized = (_blocks_0_Add_1_output_0[d0][d1][i] - mean) * inv_std;
            _ln_f_LayerNormalization_output_0[d0][d1][i] = normalized * ln_f_weight[i] + ln_f_bias[i];
        }
    }
    }
}

/*  Operator: MatMul 
    Name in model: /lm_head/MatMul
    Input: _ln_f_LayerNormalization_output_0 [1, 64, 64]
    Input: onnx_MatMul_178 [64, 6000]
    Output: _lm_head_MatMul_output_0 [1, 64, 6000]
*/
void node_lm_head_MatMul(const float _ln_f_LayerNormalization_output_0[1][64][64], const float onnx_MatMul_178[64][6000], float _lm_head_MatMul_output_0[1][64][6000]) {
    /*Matrix multiplication*/
    for (int b = 0; b < 1; b++) {
        for (int i = 0; i <  64; i++) {
            for (int j = 0; j < 6000; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 64; k++) {
                    acc += _ln_f_LayerNormalization_output_0[b][i][k] * onnx_MatMul_178[k][j];
                }
            _lm_head_MatMul_output_0[b][i][j] = acc;
            }
        }
    }
}

/*  Operator: Add 
    Name in model: /lm_head/Add
    Input: lm_head_bias [6000]
    Input: _lm_head_MatMul_output_0 [1, 64, 6000]
    Output: logits [1, 64, 6000]
*/
    // Warning: Broadcasting is applied.
void node_lm_head_Add(const float lm_head_bias[6000], const float _lm_head_MatMul_output_0[1][64][6000], float logits[1][64][6000]) {
    for (int d0 = 0; d0 < 1; d0++) {
    for (int d1 = 0; d1 < 64; d1++) {
    for (int d2 = 0; d2 < 6000; d2++) {
        logits[d0][d1][d2] = lm_head_bias[d2] + _lm_head_MatMul_output_0[d0][d1][d2];
    }
    }
    }
}

void forward_pass(const int64_t input_ids[1][64], float logits[1][64][6000])
{
    union tensor_union_0 *tu0 = (union tensor_union_0 *)ps_malloc(sizeof(union tensor_union_0));
    union tensor_union_1 *tu1 = (union tensor_union_1 *)ps_malloc(sizeof(union tensor_union_1));
    union tensor_union_2 *tu2 = (union tensor_union_2 *)ps_malloc(sizeof(union tensor_union_2));
    union tensor_union_3 *tu3 = (union tensor_union_3 *)ps_malloc(sizeof(union tensor_union_3));
    union tensor_union_4 *tu4 = (union tensor_union_4 *)ps_malloc(sizeof(union tensor_union_4));

    node_token_embedding_Gather(tensor_token_embedding_weight, input_ids, tu0->tensor_token_embedding_Gather_output_0);
    node_Add(tu0->tensor_token_embedding_Gather_output_0, tensor_onnx_Add_140, tu1->tensor_Add_output_0);
    node_blocks_0_ln1_LayerNormalization(tu1->tensor_Add_output_0, tensor_blocks_0_ln1_weight, tensor_blocks_0_ln1_bias, tu0->tensor_blocks_0_ln1_LayerNormalization_output_0);
    node_MatMul_3(tu0->tensor_blocks_0_ln1_LayerNormalization_output_0, tensor__v_92, tu2->tensor_v_93);
    node_Split_4(tu2->tensor_v_93, tensor__v_97, tu0->tensor_v_94, tu3->tensor_v_95, tu4->tensor_v_96);
    node_blocks_0_attn_key_Add(tensor_blocks_0_attn_key_bias, tu0->tensor_v_94, tu2->tensor_blocks_0_attn_key_Add_output_0);
    node_blocks_0_attn_Reshape(tu2->tensor_blocks_0_attn_key_Add_output_0, tensor__blocks_0_attn_Constant_output_0, tu0->tensor_blocks_0_attn_Reshape_output_0);
    node_blocks_0_attn_query_Add(tensor_blocks_0_attn_query_bias, tu3->tensor_v_95, tu2->tensor_blocks_0_attn_query_Add_output_0);
    node_blocks_0_attn_Reshape_1(tu2->tensor_blocks_0_attn_query_Add_output_0, tensor__blocks_0_attn_Constant_output_0, tu3->tensor_blocks_0_attn_Reshape_1_output_0);
    node_blocks_0_attn_Transpose(tu3->tensor_blocks_0_attn_Reshape_1_output_0, tu2->tensor_blocks_0_attn_Transpose_output_0);
    node_blocks_0_attn_value_Add(tensor_blocks_0_attn_value_bias, tu4->tensor_v_96, tu3->tensor_blocks_0_attn_value_Add_output_0);
    node_blocks_0_attn_Reshape_2(tu3->tensor_blocks_0_attn_value_Add_output_0, tensor__blocks_0_attn_Constant_output_0, tu4->tensor_blocks_0_attn_Reshape_2_output_0);
    node_blocks_0_attn_Transpose_1(tu4->tensor_blocks_0_attn_Reshape_2_output_0, tu3->tensor_blocks_0_attn_Transpose_1_output_0);
    node_blocks_0_attn_Transpose_2(tu0->tensor_blocks_0_attn_Reshape_output_0, tu4->tensor_blocks_0_attn_Transpose_2_output_0);
    node_blocks_0_attn_MatMul(tu2->tensor_blocks_0_attn_Transpose_output_0, tu4->tensor_blocks_0_attn_Transpose_2_output_0, tu0->tensor_blocks_0_attn_MatMul_output_0);
    node_blocks_0_attn_Mul(tu0->tensor_blocks_0_attn_MatMul_output_0, tensor__blocks_0_attn_Constant_3_output_0, tu2->tensor_blocks_0_attn_Mul_output_0);
    node_blocks_0_attn_Where(tensor_onnx_Where_170, tensor__blocks_0_attn_Constant_4_output_0, tu2->tensor_blocks_0_attn_Mul_output_0, tu0->tensor_blocks_0_attn_Where_output_0);
    node_blocks_0_attn_Softmax(tu0->tensor_blocks_0_attn_Where_output_0, tu2->tensor_blocks_0_attn_Softmax_output_0);
    node_blocks_0_attn_MatMul_1(tu2->tensor_blocks_0_attn_Softmax_output_0, tu3->tensor_blocks_0_attn_Transpose_1_output_0, tu0->tensor_blocks_0_attn_MatMul_1_output_0);
    node_blocks_0_attn_Transpose_3(tu0->tensor_blocks_0_attn_MatMul_1_output_0, tu2->tensor_blocks_0_attn_Transpose_3_output_0);
    node_blocks_0_attn_Reshape_3(tu2->tensor_blocks_0_attn_Transpose_3_output_0, tensor__blocks_0_attn_Constant_5_output_0, tu0->tensor_blocks_0_attn_Reshape_3_output_0);
    node_blocks_0_attn_proj_MatMul(tu0->tensor_blocks_0_attn_Reshape_3_output_0, tensor_onnx_MatMul_175, tu2->tensor_blocks_0_attn_proj_MatMul_output_0);
    node_blocks_0_attn_proj_Add(tensor_blocks_0_attn_proj_bias, tu2->tensor_blocks_0_attn_proj_MatMul_output_0, tu0->tensor_blocks_0_attn_proj_Add_output_0);
    node_blocks_0_Add(tu1->tensor_Add_output_0, tu0->tensor_blocks_0_attn_proj_Add_output_0, tu2->tensor_blocks_0_Add_output_0);
    node_blocks_0_ln2_LayerNormalization(tu2->tensor_blocks_0_Add_output_0, tensor_blocks_0_ln2_weight, tensor_blocks_0_ln2_bias, tu0->tensor_blocks_0_ln2_LayerNormalization_output_0);
    node_blocks_0_ffn_ffn_0_MatMul(tu0->tensor_blocks_0_ln2_LayerNormalization_output_0, tensor_onnx_MatMul_176, tu1->tensor_blocks_0_ffn_ffn_0_MatMul_output_0);
    node_blocks_0_ffn_ffn_0_Add(tensor_blocks_0_ffn_0_bias, tu1->tensor_blocks_0_ffn_ffn_0_MatMul_output_0, tu0->tensor_blocks_0_ffn_ffn_0_Add_output_0);
    node_blocks_0_ffn_ffn_1_Div(tu0->tensor_blocks_0_ffn_ffn_0_Add_output_0, tensor__blocks_0_ffn_ffn_1_Constant_output_0, tu1->tensor_blocks_0_ffn_ffn_1_Div_output_0);
    node_blocks_0_ffn_ffn_1_Erf(tu1->tensor_blocks_0_ffn_ffn_1_Div_output_0, tu3->tensor_blocks_0_ffn_ffn_1_Erf_output_0);
    node_blocks_0_ffn_ffn_1_Add(tu3->tensor_blocks_0_ffn_ffn_1_Erf_output_0, tensor__blocks_0_ffn_ffn_1_Constant_1_output_0, tu1->tensor_blocks_0_ffn_ffn_1_Add_output_0);
    node_blocks_0_ffn_ffn_1_Mul(tu0->tensor_blocks_0_ffn_ffn_0_Add_output_0, tu1->tensor_blocks_0_ffn_ffn_1_Add_output_0, tu3->tensor_blocks_0_ffn_ffn_1_Mul_output_0);
    node_blocks_0_ffn_ffn_1_Mul_1(tu3->tensor_blocks_0_ffn_ffn_1_Mul_output_0, tensor__blocks_0_ffn_ffn_1_Constant_2_output_0, tu0->tensor_blocks_0_ffn_ffn_1_Mul_1_output_0);
    node_blocks_0_ffn_ffn_2_MatMul(tu0->tensor_blocks_0_ffn_ffn_1_Mul_1_output_0, tensor_onnx_MatMul_177, tu1->tensor_blocks_0_ffn_ffn_2_MatMul_output_0);
    node_blocks_0_ffn_ffn_2_Add(tensor_blocks_0_ffn_2_bias, tu1->tensor_blocks_0_ffn_ffn_2_MatMul_output_0, tu0->tensor_blocks_0_ffn_ffn_2_Add_output_0);
    node_blocks_0_Add_1(tu2->tensor_blocks_0_Add_output_0, tu0->tensor_blocks_0_ffn_ffn_2_Add_output_0, tu1->tensor_blocks_0_Add_1_output_0);
    node_ln_f_LayerNormalization(tu1->tensor_blocks_0_Add_1_output_0, tensor_ln_f_weight, tensor_ln_f_bias, tu0->tensor_ln_f_LayerNormalization_output_0);
    node_lm_head_MatMul(tu0->tensor_ln_f_LayerNormalization_output_0, tensor_onnx_MatMul_178, tu1->tensor_lm_head_MatMul_output_0);
    node_lm_head_Add(tensor_lm_head_bias, tu1->tensor_lm_head_MatMul_output_0, logits);
    free(tu0);
    free(tu1);
    free(tu2);
    free(tu3);
    free(tu4);
}
