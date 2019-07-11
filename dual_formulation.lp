\ Model dynamicmatch_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4 dual[0] + 4 dual[1] + 4 dual[2] + 4 dual[3] + 4 dual[4] + 4 dual[5]
   + dual[6] + dual[7] + dual[8] + dual[9] + dual[10] + dual[11] + dual[12]
   + dual[13] + dual[14] + dual[15]
Subject To
 R0: - dual[0] <= -1e-05
 R1: - dual[1] <= -1e-05
 R2: - dual[2] <= -1e-05
 R3: - dual[3] <= 1
 R4: - dual[4] <= 1
 R5: - dual[5] <= 1
 R6: - dual[0] <= 1
 R7: - dual[1] <= -1e-05
 R8: - dual[2] <= -1e-05
 R9: - dual[3] <= -1e-05
 R10: - dual[4] <= 1
 R11: - dual[5] <= 1
 R12: - dual[0] <= 1
 R13: - dual[1] <= 1
 R14: - dual[2] <= -1e-05
 R15: - dual[3] <= -1e-05
 R16: - dual[4] <= -1e-05
 R17: - dual[5] <= 1
 R18: - dual[0] <= 1
 R19: - dual[1] <= 1
 R20: - dual[2] <= 1
 R21: - dual[3] <= -1e-05
 R22: - dual[4] <= -1e-05
 R23: - dual[5] <= -1e-05
 R24: - dual[6] - dual[12] <= -0.2
 R25: - dual[6] - dual[7] - dual[12] <= -0.2
 R26: - dual[6] - dual[7] - dual[8] - dual[12] <= -0.2
 R27: - dual[7] - dual[8] - dual[9] - dual[12] <= 1
 R28: - dual[8] - dual[9] - dual[10] - dual[12] <= 1
 R29: - dual[9] - dual[10] - dual[11] - dual[12] <= 1
 R30: - dual[6] - dual[13] <= 1
 R31: - dual[6] - dual[7] - dual[13] <= -0.28
 R32: - dual[6] - dual[7] - dual[8] - dual[13] <= -0.28
 R33: - dual[7] - dual[8] - dual[9] - dual[13] <= -0.28
 R34: - dual[8] - dual[9] - dual[10] - dual[13] <= 1
 R35: - dual[9] - dual[10] - dual[11] - dual[13] <= 1
 R36: - dual[6] - dual[14] <= 1
 R37: - dual[6] - dual[7] - dual[14] <= 1
 R38: - dual[6] - dual[7] - dual[8] - dual[14] <= -0.7
 R39: - dual[7] - dual[8] - dual[9] - dual[14] <= -0.7
 R40: - dual[8] - dual[9] - dual[10] - dual[14] <= -0.7
 R41: - dual[9] - dual[10] - dual[11] - dual[14] <= 1
 R42: - dual[6] - dual[15] <= 1
 R43: - dual[6] - dual[7] - dual[15] <= 1
 R44: - dual[6] - dual[7] - dual[8] - dual[15] <= 1
 R45: - dual[7] - dual[8] - dual[9] - dual[15] <= -0.3
 R46: - dual[8] - dual[9] - dual[10] - dual[15] <= -0.3
 R47: - dual[9] - dual[10] - dual[11] - dual[15] <= -0.3
Bounds
End
