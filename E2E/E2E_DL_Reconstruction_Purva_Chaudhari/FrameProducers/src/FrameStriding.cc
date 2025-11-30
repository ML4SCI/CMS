#include "RecoE2E/FrameProducers/interface/FrameStriding.h"

// Striding input frames (vDetFrame by rowstrides and colstrides accordingly)
e2e::Frame2D frameStriding(e2e::Frame2D& vDetFrame, int rows, int columns, int rowstrides, int colstrides){
  e2e::Frame2D vStridedFrame ((rows*rowstrides),e2e::Frame1D((columns*colstrides),0));
  for (int rowidx=0; rowidx<rows; rowidx++){
    for (int colidx=0; colidx<columns; colidx++){
      for (int kernelrow=0; kernelrow<rowstrides; kernelrow++){
        for (int kernelcol=0; kernelcol<colstrides; kernelcol++){
          vStridedFrame[(rowstrides*rowidx+kernelrow)][colstrides*colidx+kernelcol] = vDetFrame[rowidx][colidx]/(rowstrides*colstrides);
          //if(rowidx<5 && colidx<5) std::cout<<"("<<rowstrides*rowidx+kernelrow<<","<<colstrides*colidx+kernelcol<<"): "<<vStridedFrame[(rowstrides*rowidx+kernelrow)*columns*colstrides+colstrides*colidx+kernelcol]<<" "<<vDetFrame[rowidx*columns+colidx]/(rowstrides*colstrides);
        }
      }
    }
  }
  return vStridedFrame;
}
