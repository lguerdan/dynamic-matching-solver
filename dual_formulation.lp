\ Model dynamicmatch_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4 dual[0] + 4 dual[1] + 4 dual[2] + 4 dual[3] + 4 dual[4] + 4 dual[5]
   + dual[6] + dual[7] + dual[8] + dual[9] + dual[10] + dual[11] + dual[12]
   + dual[13] + dual[14] + dual[15]
Subject To
 R0: dual[0] + dual[12] >= 1e-05
 R1: dual[1] + dual[12] >= 1e-05
 R2: dual[2] + dual[12] >= 1e-05
 R3: dual[1] + dual[13] >= 1e-05
 R4: dual[2] + dual[13] >= 1e-05
 R5: dual[3] + dual[13] >= 1e-05
 R6: dual[2] + dual[14] >= 1e-05
 R7: dual[3] + dual[14] >= 1e-05
 R8: dual[4] + dual[14] >= 1e-05
 R9: dual[3] + dual[15] >= 1e-05
 R10: dual[4] + dual[15] >= 1e-05
 R11: dual[5] + dual[15] >= 1e-05
 R12: dual[6] + dual[12] >= 0.49
 R13: dual[6] + dual[7] + dual[12] >= 0.49
 R14: dual[6] + dual[7] + dual[8] + dual[12] >= 0.49
 R15: dual[6] + dual[7] + dual[13] >= 0.48
 R16: dual[6] + dual[7] + dual[8] + dual[13] >= 0.48
 R17: dual[6] + dual[7] + dual[8] + dual[9] + dual[13] >= 0.48
 R18: dual[6] + dual[7] + dual[8] + dual[14] >= 0.53
 R19: dual[6] + dual[7] + dual[8] + dual[9] + dual[14] >= 0.53
 R20: dual[7] + dual[8] + dual[9] + dual[10] + dual[14] >= 0.53
 R21: dual[6] + dual[7] + dual[8] + dual[9] + dual[15] >= 0.79
 R22: dual[7] + dual[8] + dual[9] + dual[10] + dual[15] >= 0.79
 R23: dual[8] + dual[9] + dual[10] + dual[11] + dual[15] >= 0.79
Bounds
End
