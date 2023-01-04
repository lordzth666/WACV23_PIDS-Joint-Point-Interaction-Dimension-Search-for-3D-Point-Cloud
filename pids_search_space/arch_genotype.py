# ---------------Baselines----------------------------------
block_args_mobilenetv2_kpcnn_no_attn = [
    "k7_e1_s1_r1_c16_d0_a0",
    "k7_e3_s2_r2_c24_d0_a0",
    "k7_e3_s2_r3_c32_d0_a0",
    "k7_e3_s2_r4_c64_d0_a0",
    "k7_e3_s1_r3_c96_d0_a0",
    "k7_e3_s2_r3_c160_d0_a0",
    "k7_e3_s1_r1_c320_d0_a0",
]

block_args_mobilenetv2_kpcnn_attn = [
    "k7_e1_s1_r1_c16_d0_a0",
    "k7_e3_s2_r2_c24_d0_a1",
    "k7_e3_s2_r3_c32_d0_a1",
    "k7_e3_s2_r4_c64_d0_a1",
    "k7_e3_s1_r3_c96_d0_a1",
    "k7_e3_s2_r3_c160_d0_a1",
    "k7_e3_s1_r1_c320_d0_a1",
]

block_args_mobilenetv2_kpfcn_no_attn = [
    "k7_e1_s1_r1_c16_d0_a0",
    "k7_e3_s2_r2_c24_d0_a0",
    "k7_e3_s2_r3_c32_d0_a0",
    "k7_e3_s2_r4_c64_d0_a0",
    "k7_e3_s1_r3_c96_d0_a0",
    "k7_e3_s2_r3_c160_d0_a0",
    "k7_e3_s1_r1_c320_d0_a0",
    "k7_e1_s1_r1_c96_d0_a0",
    "k7_e1_s1_r1_c64_d0_a0",
    "k7_e1_s1_r1_c32_d0_a0",
    "k7_e1_s1_r1_c24_d0_a0"
]

block_args_mobilenetv2_kpfcn_attn = [
    "k7_e1_s1_r1_c16_d0_a0",
    "k7_e3_s2_r2_c24_d0_a1",
    "k7_e3_s2_r3_c32_d0_a1",
    "k7_e3_s2_r4_c64_d0_a1",
    "k7_e3_s1_r3_c96_d0_a1",
    "k7_e3_s2_r3_c160_d0_a1",
    "k7_e3_s1_r1_c320_d0_a1",
    "k7_e1_s1_r1_c96_d0_a1",
    "k7_e1_s1_r1_c64_d0_a1",
    "k7_e1_s1_r1_c32_d0_a1",
    "k7_e1_s1_r1_c24_d0_a1"
]

#---------------Random Search Baseline----------------------------
block_args_kpfcn_random = {
    'best1': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e2_s2_r3_c24_d0_a0", "k5_e2_s2_r4_c32_d0_a1",
        "k7_e3_s2_r4_c40_d0_a0", "k13_e4_s1_r4_c40_d0_a1", "k13_e2_s2_r5_c64_d0_a0",
        "k7_e3_s1_r1_c160_d0_a0", "k7_e3_s1_r1_c80_d0_a0", "k7_e2_s1_r1_c56_d0_a1",
        "k7_e2_s1_r2_c32_d0_a0", "k5_e3_s1_r1_c24_d0_a0"],
    'best2': [
        "k7_e1_s1_r1_c16_d0_a0", "k13_e2_s2_r3_c24_d0_a0", "k7_e2_s2_r3_c32_d0_a0",
        "k7_e2_s2_r4_c24_d0_a0", "k5_e3_s1_r3_c56_d0_a0", "k13_e4_s2_r3_c80_d0_a1",
        "k13_e3_s1_r1_c160_d0_a0", "k7_e2_s1_r1_c96_d0_a0", "k5_e3_s1_r2_c56_d0_a0",
        "k7_e3_s1_r1_c40_d0_a0", "k13_e3_s1_r2_c16_d0_a0"],
    'best3': [
        "k5_e1_s1_r1_c16_d0_a0", "k13_e3_s2_r2_c16_d0_a0", "k7_e2_s2_r2_c32_d0_a0",
        "k7_e3_s2_r5_c32_d0_a0", "k7_e3_s1_r3_c72_d0_a0", "k13_e4_s2_r5_c64_d0_a0",
        "k7_e3_s1_r1_c160_d0_a0", "k13_e3_s1_r1_c96_d0_a1", "k13_e2_s1_r2_c72_d0_a0",
        "k13_e3_s1_r2_c32_d0_a0", "k5_e2_s1_r2_c24_d0_a0"],
    'best4': [
        "k7_e1_s1_r1_c16_d0_a0", "k5_e4_s2_r3_c16_d0_a0", "k7_e2_s2_r3_c32_d0_a1",
        "k5_e4_s2_r4_c40_d0_a1", "k13_e3_s1_r4_c72_d0_a1", "k5_e4_s2_r5_c64_d0_a1",
        "k13_e3_s1_r1_c160_d0_a1", "k7_e2_s1_r1_c96_d0_a1", "k13_e2_s1_r2_c40_d0_a1",
        "k5_e3_s1_r1_c40_d0_a1", "k5_e2_s1_r1_c24_d0_a0"],
    'best5': [
        "k5_e1_s1_r1_c16_d0_a0", "k5_e4_s2_r3_c24_d0_a0", "k7_e4_s2_r3_c32_d0_a0",
        "k13_e3_s2_r5_c40_d0_a0", "k7_e4_s1_r4_c40_d0_a0", "k13_e2_s2_r4_c80_d0_a1",
        "k13_e2_s1_r1_c160_d0_a0", "k13_e3_s1_r2_c80_d0_a1", "k7_e3_s1_r1_c56_d0_a1",
        "k13_e3_s1_r1_c40_d0_a1", "k7_e2_s1_r1_c16_d0_a0"],
}

block_args_kpcnn_random = {
    'best1': [
        "k5_e1_s1_r1_c16_d0_a0", "k7_e4_s2_r3_c16_d0_a0", "k7_e3_s2_r2_c32_d0_a1",
        "k7_e2_s2_r4_c40_d0_a1", "k5_e4_s1_r4_c40_d0_a0", "k13_e3_s2_r4_c96_d0_a1",
        "k5_e3_s1_r1_c160_d0_a1"],
    'best2': [
        "k13_e1_s1_r1_c16_d0_a0", "k7_e3_s2_r2_c16_d0_a0", "k5_e4_s2_r4_c32_d0_a0",
        "k7_e3_s2_r4_c40_d0_a0", "k13_e4_s1_r4_c40_d0_a0", "k7_e4_s2_r4_c80_d0_a0",
        "k13_e4_s1_r1_c160_d0_a0"],
    'best3': [
        "k5_e1_s1_r1_c16_d0_a0", "k5_e4_s2_r3_c24_d0_a0", "k7_e4_s2_r4_c32_d0_a0",
        "k7_e2_s2_r3_c24_d0_a0", "k13_e4_s1_r2_c56_d0_a0", "k13_e2_s2_r5_c80_d0_a1",
        "k5_e4_s1_r1_c160_d0_a0"],
    'best4': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e3_s2_r2_c16_d0_a0", "k7_e4_s2_r3_c24_d0_a1",
        "k13_e2_s2_r3_c40_d0_a1", "k13_e3_s1_r4_c40_d0_a1", "k13_e4_s2_r4_c96_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0"],
    'best5': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e2_s2_r2_c16_d0_a0", "k5_e4_s2_r3_c32_d0_a0",
        "k13_e4_s2_r3_c40_d0_a0", "k7_e2_s1_r3_c72_d0_a0", "k13_e4_s2_r3_c64_d0_a1",
        "k13_e4_s1_r1_c160_d0_a1"],
}

#--------------Searched Models: NN Embedding Pred-----------------
block_args_kpfcn_embedding_nn_predictor_regularized_ea = {
    'best1': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r3_c32_d0_a0",
        "k5_e4.00_s2_r4_c40_d0_a1", "k13_e4.00_s1_r3_c72_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e3.00_s1_r1_c96_d0_a0", "k13_e3.00_s1_r2_c72_d0_a1",
        "k7_e3.00_s1_r1_c40_d0_a0", "k5_e3.00_s1_r1_c24_d0_a0"],
    'best2': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r3_c32_d0_a1",
        "k5_e4.00_s2_r4_c40_d0_a1", "k13_e4.00_s1_r3_c72_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e3.00_s1_r1_c96_d0_a0", "k13_e3.00_s1_r2_c72_d0_a1",
        "k7_e3.00_s1_r1_c40_d0_a0", "k5_e3.00_s1_r1_c24_d0_a0"],
    'best3': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r3_c32_d0_a1",
        "k5_e4.00_s2_r4_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e3.00_s1_r1_c96_d0_a0", "k13_e3.00_s1_r2_c72_d0_a1",
        "k7_e3.00_s1_r1_c40_d0_a0", "k5_e3.00_s1_r1_c24_d0_a0"],
    'best4': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r3_c32_d0_a1",
        "k13_e4.00_s2_r4_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e3.00_s1_r1_c96_d0_a0", "k13_e3.00_s1_r2_c72_d0_a1",
        "k7_e3.00_s1_r1_c40_d0_a0", "k5_e3.00_s1_r1_c24_d0_a0"],
    'best5': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r3_c32_d0_a1",
        "k13_e4.00_s2_r4_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k7_e4.00_s2_r5_c96_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e3.00_s1_r1_c96_d0_a0", "k13_e3.00_s1_r2_c72_d0_a1",
        "k7_e3.00_s1_r1_c40_d0_a0", "k5_e3.00_s1_r1_c24_d0_a0"],
}

block_args_kpcnn_embedding_nn_predictor_regularized_ea = {
    'best1': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e4.00_s2_r3_c24_d0_a0", "k13_e3.00_s2_r2_c32_d0_a1",
        "k13_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k13_e4.00_s2_r4_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best2': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c24_d0_a0", "k13_e3.00_s2_r2_c32_d0_a1",
        "k13_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k13_e4.00_s2_r4_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best3': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r4_c32_d0_a1",
        "k13_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k13_e4.00_s2_r4_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best4': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c24_d0_a0", "k13_e3.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r4_c72_d0_a0", "k13_e4.00_s2_r4_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best5': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k13_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c72_d0_a0", "k13_e4.00_s2_r4_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
}

# -------------Searched Models: Dense-Sparse Pred-----------------
block_args_kpfcn_dense_sparse_predictor_regularized_ea = {
    'best1': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r3_c24_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r4_c72_d0_a0", "k13_e4_s2_r5_c96_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e3_s1_r1_c96_d0_a0", "k13_e3_s1_r2_c72_d0_a1",
        "k7_e3_s1_r1_c40_d0_a0", "k5_e3_s1_r1_c24_d0_a0"],
    'best2': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c24_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r4_c72_d0_a0", "k13_e4_s2_r5_c96_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e3_s1_r1_c96_d0_a0", "k13_e3_s1_r2_c72_d0_a1",
        "k7_e3_s1_r1_c40_d0_a0", "k5_e3_s1_r1_c24_d0_a0"],
    'best3': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c24_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r4_c72_d0_a0", "k13_e4_s2_r5_c96_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e3_s1_r1_c96_d0_a0", "k13_e3_s1_r2_c72_d0_a1",
        "k7_e3_s1_r1_c40_d0_a0", "k5_e3_s1_r2_c24_d0_a0"],
    'best4': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c24_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r4_c72_d0_a0", "k13_e4_s2_r5_c96_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e3_s1_r1_c96_d0_a0", "k7_e3_s1_r2_c72_d0_a1",
        "k7_e3_s1_r1_c40_d0_a0", "k5_e3_s1_r1_c24_d0_a0"],
    'best5': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c24_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r3_c72_d0_a0", "k7_e4_s2_r5_c96_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e3_s1_r1_c96_d0_a0", "k13_e3_s1_r2_c72_d0_a1",
        "k7_e3_s1_r1_c40_d0_a0", "k5_e3_s1_r1_c24_d0_a0"],
}

block_args_kpcnn_dense_sparse_predictor_regularized_ea = {
    'best1': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best2': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e4.00_s2_r3_c24_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best3': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e4.00_s2_r3_c24_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best4': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best5': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e4.00_s2_r2_c24_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r3_c40_d0_a1", "k13_e4.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r5_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
}

# FLOPS opt 1.0
block_args_kpfcn_dense_sparse_predictor_regularized_ea_flops_opt = {
    'best1': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r3_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a1",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best2': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r3_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best3': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r3_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best4': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k13_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r3_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a1",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r2_c24_d0_a0"],
    'best5': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r5_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best6': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e3_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e3_s2_r3_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best7': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r4_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best8': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e3_s2_r3_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"],
    'best9': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r4_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r2_c24_d0_a0"],
    'best10': [
        "k13_e1_s1_r1_c16_d0_a0", "k13_e3_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
        "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r4_c64_d0_a1",
        "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
        "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r2_c24_d0_a0"],
}

block_args_kpcnn_dense_sparse_predictor_regularized_ea_flops_opt = {
    'best1': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c24_d0_a1",
        "k7_e3.00_s2_r3_c40_d0_a1", "k13_e2.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best2': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c24_d0_a1",
        "k7_e3.00_s2_r3_c40_d0_a1", "k13_e2.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best3': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c24_d0_a1",
        "k7_e2.00_s2_r3_c40_d0_a1", "k13_e3.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best4': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c24_d0_a1",
        "k7_e2.00_s2_r3_c40_d0_a1", "k13_e3.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best5': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c24_d0_a0",
        "k7_e3.00_s2_r3_c40_d0_a1", "k13_e2.00_s1_r2_c40_d0_a0", "k13_e4.00_s2_r3_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"]
}

# -------------Searched Models: Dense-Sparse Pred Enlarge-----------------
block_args_kpfcn_dense_sparse_predictor_regularized_ea_flops_opt_enlarge = {
    'best1': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c16_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k13_e3.75_s2_r2_c40_d0_a1", "k13_e3.50_s1_r1_c40_d0_a0", "k13_e4.00_s2_r6_c64_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e3.00_s1_r1_c64_d0_a0", "k13_e3.00_s1_r1_c40_d0_a1",
        "k7_e3.00_s1_r1_c28_d0_a0", "k5_e2.00_s1_r1_c24_d0_a0"],
    'best2': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c16_d0_a0", "k13_e4.00_s2_r2_c32_d0_a1",
        "k13_e3.75_s2_r2_c40_d0_a1", "k13_e3.50_s1_r1_c40_d0_a0", "k13_e4.00_s2_r6_c64_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e2.75_s1_r1_c64_d0_a0", "k13_e3.00_s1_r1_c40_d0_a1",
        "k7_e3.00_s1_r1_c28_d0_a0", "k5_e2.00_s1_r1_c24_d0_a0"],
    'best3': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c16_d0_a0", "k13_e3.75_s2_r2_c32_d0_a1",
        "k13_e3.75_s2_r2_c40_d0_a1", "k13_e3.50_s1_r1_c40_d0_a0", "k13_e4.00_s2_r6_c64_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e2.75_s1_r1_c64_d0_a0", "k13_e3.00_s1_r1_c40_d0_a1",
        "k7_e3.00_s1_r1_c28_d0_a0", "k5_e2.00_s1_r1_c24_d0_a0"],
    'best4': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c16_d0_a0", "k13_e3.75_s2_r2_c32_d0_a1",
        "k13_e3.75_s2_r2_c40_d0_a1", "k13_e3.50_s1_r1_c40_d0_a0", "k13_e4.00_s2_r6_c64_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e2.50_s1_r1_c64_d0_a0", "k13_e3.00_s1_r1_c40_d0_a1",
        "k7_e3.00_s1_r1_c28_d0_a0", "k5_e2.00_s1_r1_c24_d0_a0"],
    'best5': [
        "k13_e1.00_s1_r1_c16_d0_a0", "k13_e4.00_s2_r2_c16_d0_a0", "k13_e3.75_s2_r2_c32_d0_a1",
        "k13_e3.75_s2_r2_c40_d0_a1", "k13_e3.50_s1_r1_c40_d0_a0", "k13_e4.00_s2_r6_c64_d0_a1",
        "k7_e4.00_s1_r1_c160_d0_a0", "k13_e2.50_s1_r1_c64_d0_a0", "k13_e3.00_s1_r1_c40_d0_a1",
        "k7_e3.00_s1_r1_c24_d0_a0", "k5_e2.00_s1_r1_c24_d0_a0"],
}

block_args_kpcnn_dense_sparse_predictor_regularized_ea_flops_opt_enlarge = {
    'best1': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r2_c40_d0_a1", "k13_e2.50_s1_r1_c40_d0_a0", "k13_e4.00_s2_r2_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best2': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c32_d0_a1",
        "k7_e4.00_s2_r2_c40_d0_a1", "k13_e2.75_s1_r1_c40_d0_a0", "k13_e4.00_s2_r2_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best3': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c32_d0_a1",
        "k7_e3.25_s2_r2_c40_d0_a1", "k13_e3.00_s1_r1_c40_d0_a0", "k13_e4.00_s2_r2_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
    'best4': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c32_d0_a1",
        "k7_e3.25_s2_r2_c40_d0_a1", "k13_e3.00_s1_r1_c40_d0_a0", "k13_e4.00_s2_r2_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a0"],
    'best5': [
        "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.50_s2_r2_c28_d0_a1",
        "k7_e3.25_s2_r2_c40_d0_a1", "k13_e3.00_s1_r1_c40_d0_a0", "k13_e4.00_s2_r2_c96_d0_a1",
        "k13_e4.00_s1_r1_c160_d0_a1"],
}

#--------------Best Models------------------------------------------------
# Adapted from enlarge.
block_args_golden_modelnet40 = [
    "k7_e1.00_s1_r1_c16_d0_a0", "k5_e2.00_s2_r2_c16_d0_a0", "k13_e2.00_s2_r2_c32_d0_a1",
    "k7_e4.00_s2_r2_c40_d0_a1", "k13_e3.00_s1_r1_c40_d0_a0", "k13_e4.00_s2_r2_c96_d0_a1",
    "k13_e4.00_s1_r1_c160_d0_a1"
]

block_args_golden_semantickitti = [
    "k13_e1_s1_r1_c16_d0_a0", "k13_e4_s2_r2_c16_d0_a0", "k13_e4_s2_r2_c32_d0_a1",
    "k13_e4_s2_r3_c40_d0_a1", "k7_e4_s1_r2_c40_d0_a0", "k13_e2_s2_r3_c64_d0_a1",
    "k7_e4_s1_r1_c160_d0_a0", "k13_e2_s1_r1_c64_d0_a0", "k13_e2_s1_r1_c40_d0_a0",
    "k7_e2_s1_r1_c24_d0_a0", "k5_e2_s1_r1_c24_d0_a0"
]

# Scale up for final battle. 13->15->19
block_args_golden_semantickitti_xlarge = [
    "k19_e1_s1_r1_c16_d0_a0", "k19_e4_s2_r2_c16_d0_a0", "k19_e4_s2_r2_c32_d0_a1",
    "k19_e4_s2_r3_c40_d0_a1", "k15_e4_s1_r2_c40_d0_a0", "k19_e2_s2_r3_c64_d0_a1",
    "k15_e4_s1_r1_c160_d0_a0", "k19_e2_s1_r1_c64_d0_a0", "k19_e2_s1_r1_c40_d0_a0",
    "k15_e2_s1_r1_c24_d0_a0", "k13_e2_s1_r1_c24_d0_a0"
]

# 19->21->25
block_args_golden_semantickitti_2xlarge = [
    "k25_e1_s1_r1_c16_d0_a0", "k25_e4_s2_r2_c16_d0_a0", "k25_e4_s2_r2_c32_d0_a1",
    "k25_e4_s2_r3_c40_d0_a1", "k21_e4_s1_r2_c40_d0_a0", "k25_e2_s2_r3_c64_d0_a1",
    "k21_e4_s1_r1_c160_d0_a0", "k25_e2_s1_r1_c64_d0_a0", "k25_e2_s1_r1_c40_d0_a0",
    "k21_e2_s1_r1_c24_d0_a0", "k19_e2_s1_r1_c24_d0_a0"
]