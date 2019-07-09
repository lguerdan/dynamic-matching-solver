\ Model dynamicmatch_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  6 dual[0] + 6 dual[1] + 6 dual[2] + 6 dual[3] + 6 dual[4] + 6 dual[5]
   + 6 dual[6] + 6 dual[7] + dual[8] + dual[9] + dual[10] + dual[11]
   + dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21]
Subject To
 R0: - dual[0] - dual[16] <= 0
 R1: - dual[1] - dual[16] <= 0
 R2: - dual[2] - dual[16] <= 0
 R3: - dual[3] - dual[16] <= 1
 R4: - dual[4] - dual[16] <= 1
 R5: - dual[5] - dual[16] <= 1
 R6: - dual[6] - dual[16] <= 1
 R7: - dual[7] - dual[16] <= 1
 R8: - dual[0] - dual[17] <= 1
 R9: - dual[1] - dual[17] <= 0
 R10: - dual[2] - dual[17] <= 0
 R11: - dual[3] - dual[17] <= 0
 R12: - dual[4] - dual[17] <= 1
 R13: - dual[5] - dual[17] <= 1
 R14: - dual[6] - dual[17] <= 1
 R15: - dual[7] - dual[17] <= 1
 R16: - dual[0] - dual[18] <= 1
 R17: - dual[1] - dual[18] <= 1
 R18: - dual[2] - dual[18] <= 0
 R19: - dual[3] - dual[18] <= 0
 R20: - dual[4] - dual[18] <= 0
 R21: - dual[5] - dual[18] <= 1
 R22: - dual[6] - dual[18] <= 1
 R23: - dual[7] - dual[18] <= 1
 R24: - dual[0] - dual[19] <= 1
 R25: - dual[1] - dual[19] <= 1
 R26: - dual[2] - dual[19] <= 1
 R27: - dual[3] - dual[19] <= 0
 R28: - dual[4] - dual[19] <= 0
 R29: - dual[5] - dual[19] <= 0
 R30: - dual[6] - dual[19] <= 1
 R31: - dual[7] - dual[19] <= 1
 R32: - dual[0] - dual[20] <= 1
 R33: - dual[1] - dual[20] <= 1
 R34: - dual[2] - dual[20] <= 1
 R35: - dual[3] - dual[20] <= 1
 R36: - dual[4] - dual[20] <= 0
 R37: - dual[5] - dual[20] <= 0
 R38: - dual[6] - dual[20] <= 0
 R39: - dual[7] - dual[20] <= 1
 R40: - dual[0] - dual[21] <= 1
 R41: - dual[1] - dual[21] <= 1
 R42: - dual[2] - dual[21] <= 1
 R43: - dual[3] - dual[21] <= 1
 R44: - dual[4] - dual[21] <= 1
 R45: - dual[5] - dual[21] <= 0
 R46: - dual[6] - dual[21] <= 0
 R47: - dual[7] - dual[21] <= 0
 R48: - dual[8] - dual[16] <= -0.64078
 R49: - dual[8] - dual[9] - dual[16] <= -0.64078
 R50: - dual[8] - dual[9] - dual[10] - dual[16] <= -0.64078
 R51: - dual[9] - dual[10] - dual[11] - dual[16] <= 1
 R52: - dual[10] - dual[11] - dual[12] - dual[16] <= 1
 R53: - dual[11] - dual[12] - dual[13] - dual[16] <= 1
 R54: - dual[12] - dual[13] - dual[14] - dual[16] <= 1
 R55: - dual[13] - dual[14] - dual[15] - dual[16] <= 1
 R56: - dual[8] - dual[17] <= 1
 R57: - dual[8] - dual[9] - dual[17] <= -0.56778
 R58: - dual[8] - dual[9] - dual[10] - dual[17] <= -0.56778
 R59: - dual[9] - dual[10] - dual[11] - dual[17] <= -0.56778
 R60: - dual[10] - dual[11] - dual[12] - dual[17] <= 1
 R61: - dual[11] - dual[12] - dual[13] - dual[17] <= 1
 R62: - dual[12] - dual[13] - dual[14] - dual[17] <= 1
 R63: - dual[13] - dual[14] - dual[15] - dual[17] <= 1
 R64: - dual[8] - dual[18] <= 1
 R65: - dual[8] - dual[9] - dual[18] <= 1
 R66: - dual[8] - dual[9] - dual[10] - dual[18] <= -0.74014
 R67: - dual[9] - dual[10] - dual[11] - dual[18] <= -0.74014
 R68: - dual[10] - dual[11] - dual[12] - dual[18] <= -0.74014
 R69: - dual[11] - dual[12] - dual[13] - dual[18] <= 1
 R70: - dual[12] - dual[13] - dual[14] - dual[18] <= 1
 R71: - dual[13] - dual[14] - dual[15] - dual[18] <= 1
 R72: - dual[8] - dual[19] <= 1
 R73: - dual[8] - dual[9] - dual[19] <= 1
 R74: - dual[8] - dual[9] - dual[10] - dual[19] <= 1
 R75: - dual[9] - dual[10] - dual[11] - dual[19] <= -0.62447
 R76: - dual[10] - dual[11] - dual[12] - dual[19] <= -0.62447
 R77: - dual[11] - dual[12] - dual[13] - dual[19] <= -0.62447
 R78: - dual[12] - dual[13] - dual[14] - dual[19] <= 1
 R79: - dual[13] - dual[14] - dual[15] - dual[19] <= 1
 R80: - dual[8] - dual[20] <= 1
 R81: - dual[8] - dual[9] - dual[20] <= 1
 R82: - dual[8] - dual[9] - dual[10] - dual[20] <= 1
 R83: - dual[9] - dual[10] - dual[11] - dual[20] <= 1
 R84: - dual[10] - dual[11] - dual[12] - dual[20] <= -0.78189
 R85: - dual[11] - dual[12] - dual[13] - dual[20] <= -0.78189
 R86: - dual[12] - dual[13] - dual[14] - dual[20] <= -0.78189
 R87: - dual[13] - dual[14] - dual[15] - dual[20] <= 1
 R88: - dual[8] - dual[21] <= 1
 R89: - dual[8] - dual[9] - dual[21] <= 1
 R90: - dual[8] - dual[9] - dual[10] - dual[21] <= 1
 R91: - dual[9] - dual[10] - dual[11] - dual[21] <= 1
 R92: - dual[10] - dual[11] - dual[12] - dual[21] <= 1
 R93: - dual[11] - dual[12] - dual[13] - dual[21] <= -0.80111
 R94: - dual[12] - dual[13] - dual[14] - dual[21] <= -0.80111
 R95: - dual[13] - dual[14] - dual[15] - dual[21] <= -0.80111
Bounds
End
