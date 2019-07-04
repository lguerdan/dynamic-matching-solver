\ Model dynamicmatch_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  10 dual[0] + 10 dual[1] + 10 dual[2] + 10 dual[3] + 10 dual[4]
   + 10 dual[5] + 10 dual[6] + 10 dual[7] + 10 dual[8] + 10 dual[9]
   + 10 dual[10] + 10 dual[11] + 2 dual[12] + 2 dual[13] + 2 dual[14]
   + 2 dual[15] + 2 dual[16] + 2 dual[17] + 2 dual[18] + 2 dual[19]
   + 2 dual[20] + 2 dual[21] + 2 dual[22] + 2 dual[23] + dual[24]
   + dual[25] + dual[26] + dual[27] + dual[28] + dual[29] + dual[30]
   + dual[31] + dual[32] + dual[33] + dual[34] + dual[35] + dual[36]
   + dual[37] + dual[38] + dual[39] + dual[40] + dual[41] + dual[42]
   + dual[43] + dual[44] + dual[45]
Subject To
 R0: dual[0] + dual[36] >= 1e-05
 R1: dual[1] + dual[36] >= 1e-05
 R2: dual[2] + dual[36] >= 1e-05
 R3: dual[1] + dual[37] >= 1e-05
 R4: dual[2] + dual[37] >= 1e-05
 R5: dual[3] + dual[37] >= 1e-05
 R6: dual[2] + dual[38] >= 1e-05
 R7: dual[3] + dual[38] >= 1e-05
 R8: dual[4] + dual[38] >= 1e-05
 R9: dual[3] + dual[39] >= 1e-05
 R10: dual[4] + dual[39] >= 1e-05
 R11: dual[5] + dual[39] >= 1e-05
 R12: dual[4] + dual[40] >= 1e-05
 R13: dual[5] + dual[40] >= 1e-05
 R14: dual[6] + dual[40] >= 1e-05
 R15: dual[5] + dual[41] >= 1e-05
 R16: dual[6] + dual[41] >= 1e-05
 R17: dual[7] + dual[41] >= 1e-05
 R18: dual[6] + dual[42] >= 1e-05
 R19: dual[7] + dual[42] >= 1e-05
 R20: dual[8] + dual[42] >= 1e-05
 R21: dual[7] + dual[43] >= 1e-05
 R22: dual[8] + dual[43] >= 1e-05
 R23: dual[9] + dual[43] >= 1e-05
 R24: dual[8] + dual[44] >= 1e-05
 R25: dual[9] + dual[44] >= 1e-05
 R26: dual[10] + dual[44] >= 1e-05
 R27: dual[9] + dual[45] >= 1e-05
 R28: dual[10] + dual[45] >= 1e-05
 R29: dual[11] + dual[45] >= 1e-05
 R30: dual[12] + dual[36] >= 0.4219960671586328
 R31: dual[12] + dual[13] + dual[36] >= 0.4219960671586328
 R32: dual[12] + dual[13] + dual[14] + dual[36] >= 0.4219960671586328
 R33: dual[12] + dual[13] + dual[37] >= 0.457677810849023
 R34: dual[12] + dual[13] + dual[14] + dual[37] >= 0.457677810849023
 R35: dual[12] + dual[13] + dual[14] + dual[15] + dual[37]
   >= 0.457677810849023
 R36: dual[12] + dual[13] + dual[14] + dual[38] >= 0.9311016999243136
 R37: dual[12] + dual[13] + dual[14] + dual[15] + dual[38]
   >= 0.9311016999243136
 R38: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[38]
   >= 0.9311016999243136
 R39: dual[12] + dual[13] + dual[14] + dual[15] + dual[39]
   >= 0.7738993642332241
 R40: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[39]
   >= 0.7738993642332241
 R41: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[39] >= 0.7738993642332241
 R42: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[40]
   >= 0.5693034200864846
 R43: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[40] >= 0.5693034200864846
 R44: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[40] >= 0.5693034200864846
 R45: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[41] >= 0.4814172536542295
 R46: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[41] >= 0.4814172536542295
 R47: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[41] >= 0.4814172536542295
 R48: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[42] >= 0.2607588100621491
 R49: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[42] >= 0.2607588100621491
 R50: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[42] >= 0.2607588100621491
 R51: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[43] >= 0.0208610964349036
 R52: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[43] >= 0.0208610964349036
 R53: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21] + dual[43]
   >= 0.0208610964349036
 R54: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[44] >= 0.930654260705815
 R55: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21] + dual[44]
   >= 0.930654260705815
 R56: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21] + dual[22] + dual[44]
   >= 0.930654260705815
 R57: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21] + dual[45]
   >= 0.4452692240182915
 R58: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21] + dual[22] + dual[45]
   >= 0.4452692240182915
 R59: dual[12] + dual[13] + dual[14] + dual[15] + dual[16] + dual[17]
   + dual[18] + dual[19] + dual[20] + dual[21] + dual[22] + dual[23]
   + dual[45] >= 0.4452692240182915
 R60: dual[24] + dual[36] >= 0.1929425165380314
 R61: dual[24] + dual[25] + dual[36] >= 0.1929425165380314
 R62: dual[24] + dual[25] + dual[26] + dual[36] >= 0.1929425165380314
 R63: dual[24] + dual[25] + dual[37] >= 0.9509559985932045
 R64: dual[24] + dual[25] + dual[26] + dual[37] >= 0.9509559985932045
 R65: dual[24] + dual[25] + dual[26] + dual[27] + dual[37]
   >= 0.9509559985932045
 R66: dual[24] + dual[25] + dual[26] + dual[38] >= 0.6440431896320492
 R67: dual[24] + dual[25] + dual[26] + dual[27] + dual[38]
   >= 0.6440431896320492
 R68: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[38]
   >= 0.6440431896320492
 R69: dual[24] + dual[25] + dual[26] + dual[27] + dual[39]
   >= 0.8274919158711066
 R70: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[39]
   >= 0.8274919158711066
 R71: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[39] >= 0.8274919158711066
 R72: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[40]
   >= 0.8291995326019059
 R73: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[40] >= 0.8291995326019059
 R74: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[30] + dual[40] >= 0.8291995326019059
 R75: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[41] >= 0.7837056385545209
 R76: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[30] + dual[41] >= 0.7837056385545209
 R77: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[30] + dual[31] + dual[41] >= 0.7837056385545209
 R78: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[30] + dual[42] >= 0.1248913638581958
 R79: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[30] + dual[31] + dual[42] >= 0.1248913638581958
 R80: dual[25] + dual[26] + dual[27] + dual[28] + dual[29] + dual[30]
   + dual[31] + dual[32] + dual[42] >= 0.1248913638581958
 R81: dual[24] + dual[25] + dual[26] + dual[27] + dual[28] + dual[29]
   + dual[30] + dual[31] + dual[43] >= 0.1926068530267402
 R82: dual[25] + dual[26] + dual[27] + dual[28] + dual[29] + dual[30]
   + dual[31] + dual[32] + dual[43] >= 0.1926068530267402
 R83: dual[26] + dual[27] + dual[28] + dual[29] + dual[30] + dual[31]
   + dual[32] + dual[33] + dual[43] >= 0.1926068530267402
 R84: dual[25] + dual[26] + dual[27] + dual[28] + dual[29] + dual[30]
   + dual[31] + dual[32] + dual[44] >= 0.2367815363289603
 R85: dual[26] + dual[27] + dual[28] + dual[29] + dual[30] + dual[31]
   + dual[32] + dual[33] + dual[44] >= 0.2367815363289603
 R86: dual[27] + dual[28] + dual[29] + dual[30] + dual[31] + dual[32]
   + dual[33] + dual[34] + dual[44] >= 0.2367815363289603
 R87: dual[26] + dual[27] + dual[28] + dual[29] + dual[30] + dual[31]
   + dual[32] + dual[33] + dual[45] >= 0.3008707697598496
 R88: dual[27] + dual[28] + dual[29] + dual[30] + dual[31] + dual[32]
   + dual[33] + dual[34] + dual[45] >= 0.3008707697598496
 R89: dual[28] + dual[29] + dual[30] + dual[31] + dual[32] + dual[33]
   + dual[34] + dual[35] + dual[45] >= 0.3008707697598496
Bounds
End
