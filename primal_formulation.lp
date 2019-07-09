\ Model dynamicmatch_primal
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - primal[3] - primal[4] - primal[5] - primal[6] - primal[7] - primal[8]
   - primal[12] - primal[13] - primal[14] - primal[15] - primal[16]
   - primal[17] - primal[21] - primal[22] - primal[23] - primal[24]
   - primal[25] - primal[26] - primal[30] - primal[31] - primal[32]
   - primal[33] - primal[34] - primal[35] - primal[39] - primal[40]
   - primal[41] - primal[42] - primal[43] - primal[44] + 0.64078 primal[48]
   + 0.64078 primal[49] + 0.64078 primal[50] - primal[51] - primal[52]
   - primal[53] - primal[54] - primal[55] - primal[56] + 0.56778 primal[57]
   + 0.56778 primal[58] + 0.56778 primal[59] - primal[60] - primal[61]
   - primal[62] - primal[63] - primal[64] - primal[65] + 0.74014 primal[66]
   + 0.74014 primal[67] + 0.74014 primal[68] - primal[69] - primal[70]
   - primal[71] - primal[72] - primal[73] - primal[74] + 0.62447 primal[75]
   + 0.62447 primal[76] + 0.62447 primal[77] - primal[78] - primal[79]
   - primal[80] - primal[81] - primal[82] - primal[83] + 0.78189 primal[84]
   + 0.78189 primal[85] + 0.78189 primal[86] - primal[87] - primal[88]
   - primal[89] - primal[90] - primal[91] - primal[92] + 0.80111 primal[93]
   + 0.80111 primal[94] + 0.80111 primal[95]
Subject To
 R0: primal[0] <= 6
 R1: primal[1] + primal[9] <= 6
 R2: primal[2] + primal[10] + primal[18] <= 6
 R3: primal[11] + primal[19] + primal[27] <= 6
 R4: primal[20] + primal[28] + primal[36] <= 6
 R5: primal[29] + primal[37] + primal[45] <= 6
 R6: primal[38] + primal[46] <= 6
 R7: primal[47] <= 6
 R8: primal[48] + primal[49] + primal[50] + primal[57] + primal[58]
   + primal[66] <= 1
 R9: primal[49] + primal[50] + primal[57] + primal[58] + primal[59]
   + primal[66] + primal[67] + primal[75] <= 1
 R10: primal[50] + primal[58] + primal[59] + primal[66] + primal[67]
   + primal[68] + primal[75] + primal[76] + primal[84] <= 1
 R11: primal[59] + primal[67] + primal[68] + primal[75] + primal[76]
   + primal[77] + primal[84] + primal[85] + primal[93] <= 1
 R12: primal[68] + primal[76] + primal[77] + primal[84] + primal[85]
   + primal[86] + primal[93] + primal[94] <= 1
 R13: primal[77] + primal[85] + primal[86] + primal[93] + primal[94]
   + primal[95] <= 1
 R14: primal[86] + primal[94] + primal[95] <= 1
 R15: primal[95] <= 1
 R16: primal[0] + primal[1] + primal[2] + primal[48] + primal[49]
   + primal[50] <= 1
 R17: primal[9] + primal[10] + primal[11] + primal[57] + primal[58]
   + primal[59] <= 1
 R18: primal[18] + primal[19] + primal[20] + primal[66] + primal[67]
   + primal[68] <= 1
 R19: primal[27] + primal[28] + primal[29] + primal[75] + primal[76]
   + primal[77] <= 1
 R20: primal[36] + primal[37] + primal[38] + primal[84] + primal[85]
   + primal[86] <= 1
 R21: primal[45] + primal[46] + primal[47] + primal[93] + primal[94]
   + primal[95] <= 1
Bounds
End
