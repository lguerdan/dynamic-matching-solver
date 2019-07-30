\ Model dynamicmatch_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 dual[0] + 2 dual[1] + 2 dual[2] + 2 dual[3] + dual[4] + dual[5]
   + dual[6] + dual[7] + dual[8] + dual[9] + dual[10] + dual[11] + dual[12]
   + dual[13]
Subject To
 R0: - dual[0] <= 0
 R1: - dual[1] <= 0
 R2: - dual[2] <= 0
 R3: - dual[3] <= 1
 R4: - dual[0] <= 1
 R5: - dual[1] <= 0
 R6: - dual[2] <= 0
 R7: - dual[3] <= 0
 R8: - dual[4] - dual[12] <= -0.05
 R9: - dual[4] - dual[5] - dual[12] <= -0.05000456
 R10: - dual[5] - dual[6] - dual[12] <= -0.05003987
 R11: - dual[6] - dual[7] - dual[12] <= 1
 R12: - dual[4] - dual[13] <= 1
 R13: - dual[4] - dual[5] - dual[13] <= -0.05001831
 R14: - dual[5] - dual[6] - dual[13] <= -0.05000321
 R15: - dual[6] - dual[7] - dual[13] <= -0.05008826
 R16: - dual[8] - dual[12] <= -0.9
 R17: - dual[8] - dual[9] - dual[12] <= -0.90004443
 R18: - dual[9] - dual[10] - dual[12] <= -0.90007253
 R19: - dual[10] - dual[11] - dual[12] <= 1
 R20: - dual[8] - dual[13] <= 1
 R21: - dual[8] - dual[9] - dual[13] <= -0.90001454
 R22: - dual[9] - dual[10] - dual[13] <= -0.9000771400000001
 R23: - dual[10] - dual[11] - dual[13] <= -0.90007677
Bounds
End
