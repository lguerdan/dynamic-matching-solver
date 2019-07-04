\ Model dynamicmatch_primal
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.2500899215206653 primal[0] + 0.3992427361186728 primal[1]
   + 0.1297581890610552 primal[2] - primal[3]
   + 0.7894998009045781 primal[4] - primal[5]
   + 0.7076032298876785 primal[6] - primal[7]
   + 0.5447618150018415 primal[8] + 0.5393936176291227 primal[9]
   + 0.4607772283699725 primal[10] - primal[11]
   + 0.3367128291900694 primal[12] - primal[13]
   + 0.3160933232847112 primal[14] + 0.947688428340746 primal[15]
   + 0.2074716688313473 primal[16] + 0.6080178768004746 primal[17]
   + 0.3019940776675263 primal[18] - primal[19] - primal[20] - primal[21]
   + 0.7660873948244287 primal[22] + 0.7870569927723765 primal[23]
   - primal[24] + 0.7660873948244287 primal[25] - primal[26]
   + 0.6284529029787898 primal[27] - primal[28]
   + 0.7870569927723765 primal[29] + 0.6284529029787898 primal[30]
   - primal[31]
Subject To
 R0: primal[0] + primal[1] + primal[2] + primal[3] + primal[5] + primal[9]
   + primal[13] + primal[17] + primal[21] + primal[25] + primal[29] <= 1
 R1: primal[0] + primal[1] + primal[2] + primal[3] + primal[6] + primal[10]
   + primal[14] + primal[18] + primal[22] + primal[26] + primal[30] <= 1
 R2: primal[0] + primal[1] + primal[2] + primal[3] + primal[7] + primal[11]
   + primal[15] + primal[19] + primal[23] + primal[27] + primal[31] <= 1
 R3: primal[1] + primal[4] + primal[5] + primal[6] + primal[7] + primal[9]
   + primal[13] + primal[17] + primal[21] + primal[25] + primal[29] <= 1
 R4: primal[2] + primal[4] + primal[5] + primal[6] + primal[7] + primal[10]
   + primal[14] + primal[18] + primal[22] + primal[26] + primal[30] <= 1
 R5: primal[3] + primal[4] + primal[5] + primal[6] + primal[7] + primal[11]
   + primal[15] + primal[19] + primal[23] + primal[27] + primal[31] <= 1
 R6: primal[1] + primal[5] + primal[8] + primal[9] + primal[10]
   + primal[11] + primal[13] + primal[17] + primal[21] + primal[25]
   + primal[29] <= 1
 R7: primal[2] + primal[6] + primal[8] + primal[9] + primal[10]
   + primal[11] + primal[14] + primal[18] + primal[22] + primal[26]
   + primal[30] <= 1
 R8: primal[3] + primal[7] + primal[8] + primal[9] + primal[10]
   + primal[11] + primal[15] + primal[19] + primal[23] + primal[27]
   + primal[31] <= 1
 R9: primal[1] + primal[5] + primal[9] + primal[12] + primal[13]
   + primal[14] + primal[15] + primal[17] + primal[21] + primal[25]
   + primal[29] <= 1
 R10: primal[2] + primal[6] + primal[10] + primal[12] + primal[13]
   + primal[14] + primal[15] + primal[18] + primal[22] + primal[26]
   + primal[30] <= 1
 R11: primal[3] + primal[7] + primal[11] + primal[12] + primal[13]
   + primal[14] + primal[15] + primal[19] + primal[23] + primal[27]
   + primal[31] <= 1
 R12: primal[1] + primal[5] + primal[9] + primal[13] + primal[16]
   + primal[17] + primal[18] + primal[19] + primal[21] + primal[25]
   + primal[29] <= 1
 R13: primal[2] + primal[6] + primal[10] + primal[14] + primal[16]
   + primal[17] + primal[18] + primal[19] + primal[22] + primal[26]
   + primal[30] <= 1
 R14: primal[3] + primal[7] + primal[11] + primal[15] + primal[16]
   + primal[17] + primal[18] + primal[19] + primal[23] + primal[27]
   + primal[31] <= 1
 R15: primal[1] + primal[5] + primal[9] + primal[13] + primal[17]
   + 2 primal[21] + primal[22] + primal[23] + primal[25] + primal[29] <= 1
 R16: primal[2] + primal[6] + primal[10] + primal[14] + primal[18]
   + primal[22] + primal[25] + 2 primal[26] + primal[27] + primal[30] <= 1
 R17: primal[3] + primal[7] + primal[11] + primal[15] + primal[19]
   + primal[23] + primal[27] + primal[29] + primal[30] + 2 primal[31] <= 1
Bounds
End
