\ Model dynamicmatch_primal
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - primal[3] - primal[4] - primal[5] - primal[6] - primal[10] - primal[11]
   - primal[12] - primal[13] - primal[17] - primal[18] - primal[19]
   - primal[20] + 0.9981 primal[24] + 0.97907 primal[25]
   + 0.20727 primal[26] - primal[27] - primal[28] - primal[29] - primal[30]
   + 0.54837 primal[31] + 0.5507 primal[32] + 0.32153 primal[33]
   - primal[34] - primal[35] - primal[36] - primal[37] + 0.35709 primal[38]
   + 0.23626 primal[39] + 0.94216 primal[40] - primal[41] - primal[42]
   - primal[43] - primal[44] + 0.87939 primal[45] + 0.05081 primal[46]
   + 0.06784 primal[47]
Subject To
 R0: primal[0] + primal[6] + primal[12] + primal[18] <= 4
 R1: primal[1] + primal[7] + primal[13] + primal[19] <= 4
 R2: primal[2] + primal[8] + primal[14] + primal[20] <= 4
 R3: primal[3] + primal[9] + primal[15] + primal[21] <= 4
 R4: primal[4] + primal[10] + primal[16] + primal[22] <= 4
 R5: primal[5] + primal[11] + primal[17] + primal[23] <= 4
 R6: primal[24] + primal[25] + primal[26] + primal[27] + primal[28]
   + primal[29] + primal[30] + primal[31] + primal[32] + primal[33]
   + primal[34] + primal[35] + primal[36] + primal[37] + primal[38]
   + primal[39] + primal[40] + primal[41] + primal[42] + primal[43]
   + primal[44] + primal[45] + primal[46] + primal[47] <= 3
 R7: primal[25] + primal[26] + primal[27] + primal[28] + primal[29]
   + primal[31] + primal[32] + primal[33] + primal[34] + primal[35]
   + primal[37] + primal[38] + primal[39] + primal[40] + primal[41]
   + primal[43] + primal[44] + primal[45] + primal[46] + primal[47] <= 3
 R8: primal[26] + primal[27] + primal[28] + primal[29] + primal[32]
   + primal[33] + primal[34] + primal[35] + primal[38] + primal[39]
   + primal[40] + primal[41] + primal[44] + primal[45] + primal[46]
   + primal[47] <= 3
 R9: primal[27] + primal[28] + primal[29] + primal[33] + primal[34]
   + primal[35] + primal[39] + primal[40] + primal[41] + primal[45]
   + primal[46] + primal[47] <= 3
 R10: primal[28] + primal[29] + primal[34] + primal[35] + primal[40]
   + primal[41] + primal[46] + primal[47] <= 3
 R11: primal[29] + primal[35] + primal[41] + primal[47] <= 3
 R12: primal[24] + primal[25] + primal[26] <= 1
 R13: primal[31] + primal[32] + primal[33] <= 1
 R14: primal[38] + primal[39] + primal[40] <= 1
 R15: primal[45] + primal[46] + primal[47] <= 1
Bounds
End
