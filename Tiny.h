#ifndef TINY_H
#define TINY_H
#include <stdint.h>
#include <stdbool.h>
extern int num_tokens;
union tensor_union_0
{

    float tensor_token_embedding_Gather_output_0[1][64][64]; // 4096
    float tensor_blocks_0_ln1_LayerNormalization_output_0[1][64][64]; // 4096
    float tensor_v_94[1][64][64]; // 4096
    float tensor_blocks_0_attn_Reshape_output_0[1][64][4][16]; // 4096
    float tensor_blocks_0_attn_MatMul_output_0[1][4][64][64]; // 16384
    float tensor_blocks_0_attn_Where_output_0[1][4][64][64]; // 16384
    float tensor_blocks_0_attn_MatMul_1_output_0[1][4][64][16]; // 4096
    float tensor_blocks_0_attn_Reshape_3_output_0[1][64][64]; // 4096
    float tensor_blocks_0_attn_proj_Add_output_0[1][64][64]; // 4096
    float tensor_blocks_0_ln2_LayerNormalization_output_0[1][64][64]; // 4096
    float tensor_blocks_0_ffn_ffn_0_Add_output_0[1][64][256]; // 16384
    float tensor_blocks_0_ffn_ffn_1_Mul_1_output_0[1][64][256]; // 16384
    float tensor_blocks_0_ffn_ffn_2_Add_output_0[1][64][64]; // 4096
    float tensor_ln_f_LayerNormalization_output_0[1][64][64]; // 4096
};
static union tensor_union_0 tu0;
union tensor_union_1
{

    float tensor_Add_output_0[1][64][64]; // 4096
    float tensor_blocks_0_ffn_ffn_0_MatMul_output_0[1][64][256]; // 16384
    float tensor_blocks_0_ffn_ffn_1_Div_output_0[1][64][256]; // 16384
    float tensor_blocks_0_ffn_ffn_1_Add_output_0[1][64][256]; // 16384
    float tensor_blocks_0_ffn_ffn_2_MatMul_output_0[1][64][64]; // 4096
    float tensor_blocks_0_Add_1_output_0[1][64][64]; // 4096
    float tensor_lm_head_MatMul_output_0[1][64][6000]; // 384000
};
static union tensor_union_1 tu1;
union tensor_union_2
{

    float tensor_v_93[1][64][192]; // 12288
    float tensor_blocks_0_attn_key_Add_output_0[1][64][64]; // 4096
    float tensor_blocks_0_attn_query_Add_output_0[1][64][64]; // 4096
    float tensor_blocks_0_attn_Transpose_output_0[1][4][64][16]; // 4096
    float tensor_blocks_0_attn_Mul_output_0[1][4][64][64]; // 16384
    float tensor_blocks_0_attn_Softmax_output_0[1][4][64][64]; // 16384
    float tensor_blocks_0_attn_Transpose_3_output_0[1][64][4][16]; // 4096
    float tensor_blocks_0_attn_proj_MatMul_output_0[1][64][64]; // 4096
    float tensor_blocks_0_Add_output_0[1][64][64]; // 4096
};
static union tensor_union_2 tu2;
union tensor_union_3
{

    float tensor_v_95[1][64][64]; // 4096
    float tensor_blocks_0_attn_Reshape_1_output_0[1][64][4][16]; // 4096
    float tensor_blocks_0_attn_value_Add_output_0[1][64][64]; // 4096
    float tensor_blocks_0_attn_Transpose_1_output_0[1][4][64][16]; // 4096
    float tensor_blocks_0_ffn_ffn_1_Erf_output_0[1][64][256]; // 16384
    float tensor_blocks_0_ffn_ffn_1_Mul_output_0[1][64][256]; // 16384
};
//static union tensor_union_3 tu3;
union tensor_union_4
{

    float tensor_v_96[1][64][64]; // 4096
    float tensor_blocks_0_attn_Reshape_2_output_0[1][64][4][16]; // 4096
    float tensor_blocks_0_attn_Transpose_2_output_0[1][4][16][64]; // 4096
};
static union tensor_union_4 tu4;
void forward_pass(const int64_t input_ids[1][64], float logits[1][64][6000]);
#endif // TINY_H