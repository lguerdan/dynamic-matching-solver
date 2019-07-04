\ Model dynamicmatch_primal
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.4362549991459618 primal[0] + 7.2154822377124006e-04 primal[1]
   - primal[2] - primal[3] + 0.6414346924371566 primal[4]
   + 0.1909400736414941 primal[5] + 0.0162537090998965 primal[6]
   + 0.8764154816396729 primal[7] + 0.13244337461288 primal[8] - primal[9]
   - primal[10] + 0.8535678315745178 primal[11] - primal[12] - primal[13]
   + 0.2711537983797578 primal[14] + 0.3437609938209717 primal[15]
   - primal[16] + 0.2711537983797578 primal[17] - primal[18]
   + 0.9834459741033259 primal[19] - primal[20]
   + 0.3437609938209717 primal[21] + 0.9834459741033259 primal[22]
   - primal[23]
Subject To
 R0: primal[0] + primal[1] + primal[2] + primal[3] <= 1
 R1: primal[4] + primal[5] + primal[6] + primal[7] <= 1
 R2: primal[8] + primal[9] + primal[10] + primal[11] <= 1
 R3: primal[1] + primal[5] + primal[9] + 2 primal[13] + primal[14]
   + primal[15] + primal[17] + primal[21] <= 1
 R4: primal[2] + primal[6] + primal[10] + primal[14] + primal[17]
   + 2 primal[18] + primal[19] + primal[22] <= 1
 R5: primal[3] + primal[7] + primal[11] + primal[15] + primal[19]
   + primal[21] + primal[22] + 2 primal[23] <= 1
Bounds
End
