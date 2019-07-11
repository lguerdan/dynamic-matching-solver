\ Model dynamicmatch_primal
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  1e-05 primal[0] + 1e-05 primal[1] + 1e-05 primal[2] - primal[3]
   - primal[4] - primal[5] - primal[6] + 1e-05 primal[7] + 1e-05 primal[8]
   + 1e-05 primal[9] - primal[10] - primal[11] - primal[12] - primal[13]
   + 1e-05 primal[14] + 1e-05 primal[15] + 1e-05 primal[16] - primal[17]
   - primal[18] - primal[19] - primal[20] + 1e-05 primal[21]
   + 1e-05 primal[22] + 1e-05 primal[23] + 0.2 primal[24] + 0.2 primal[25]
   + 0.2 primal[26] - primal[27] - primal[28] - primal[29] - primal[30]
   + 0.28 primal[31] + 0.28 primal[32] + 0.28 primal[33] - primal[34]
   - primal[35] - primal[36] - primal[37] + 0.7 primal[38] + 0.7 primal[39]
   + 0.7 primal[40] - primal[41] - primal[42] - primal[43] - primal[44]
   + 0.3 primal[45] + 0.3 primal[46] + 0.3 primal[47]
Subject To
 R0: primal[0] <= 4
 R1: primal[1] + primal[7] <= 4
 R2: primal[2] + primal[8] + primal[14] <= 4
 R3: primal[9] + primal[15] + primal[21] <= 4
 R4: primal[16] + primal[22] <= 4
 R5: primal[23] <= 4
 R6: primal[24] + primal[25] + primal[26] + primal[31] + primal[32]
   + primal[38] <= 1
 R7: primal[25] + primal[26] + primal[31] + primal[32] + primal[33]
   + primal[38] + primal[39] + primal[45] <= 1
 R8: primal[26] + primal[32] + primal[33] + primal[38] + primal[39]
   + primal[40] + primal[45] + primal[46] <= 1
 R9: primal[33] + primal[39] + primal[40] + primal[45] + primal[46]
   + primal[47] <= 1
 R10: primal[40] + primal[46] + primal[47] <= 1
 R11: primal[47] <= 1
 R12: primal[24] + primal[25] + primal[26] <= 1
 R13: primal[31] + primal[32] + primal[33] <= 1
 R14: primal[38] + primal[39] + primal[40] <= 1
 R15: primal[45] + primal[46] + primal[47] <= 1
Bounds
End
