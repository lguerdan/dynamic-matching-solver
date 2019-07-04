\ Model dynamicmatch_dual
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  dual[0] + dual[1] + dual[2] + dual[3] + dual[4] + dual[5]
Subject To
 R0: - dual[0] <= -0.4362549991459618
 R1: - dual[0] - dual[3] <= -7.2154822377124006e-04
 R2: - dual[0] - dual[4] <= 1
 R3: - dual[0] - dual[5] <= 1
 R4: - dual[1] <= -0.6414346924371566
 R5: - dual[1] - dual[3] <= -0.1909400736414941
 R6: - dual[1] - dual[4] <= -0.0162537090998965
 R7: - dual[1] - dual[5] <= -0.8764154816396729
 R8: - dual[2] <= -0.13244337461288
 R9: - dual[2] - dual[3] <= 1
 R10: - dual[2] - dual[4] <= 1
 R11: - dual[2] - dual[5] <= -0.8535678315745178
 R12: - 2 dual[3] <= 1
 R13: - dual[3] - dual[4] <= -0.2711537983797578
 R14: - dual[3] - dual[5] <= -0.3437609938209717
 R15: - dual[3] - dual[4] <= -0.2711537983797578
 R16: - 2 dual[4] <= 1
 R17: - dual[4] - dual[5] <= -0.9834459741033259
 R18: - dual[3] - dual[5] <= -0.3437609938209717
 R19: - dual[4] - dual[5] <= -0.9834459741033259
 R20: - 2 dual[5] <= 1
Bounds
End
