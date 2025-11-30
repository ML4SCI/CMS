#ifndef RecoE2E_FrameProducers_h
#define RecoE2E_FrameProducers_h

#include <iostream>
#include <vector>
#include <cassert>

#include "RecoE2E/DataFormats/interface/FrameCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace e2e {

  void getFrame( e2e::Frame2D&, const e2e::seed&, const e2e::Frame2D*, int, int );

}
#endif
