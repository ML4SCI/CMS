#ifndef RecoE2E_FrameCollections_h
#define RecoE2E_FrameCollections_h

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include <vector>

namespace e2e {

  typedef float pred;
  typedef std::vector<int> seed;

  typedef std::vector<float> Frame1D;
  typedef std::vector<std::vector<float> > Frame2D;
  typedef std::vector<std::vector<std::vector<float> > > Frame3D;
  typedef std::vector<std::vector<std::vector<std::vector<float> > > > Frame4D;

  // AOD (reco::photon) collections
  typedef edm::AssociationVector<reco::PhotonRefProd, std::vector<pred> >    PhoPredCollection;
  typedef edm::AssociationVector<reco::PhotonRefProd, std::vector<seed> >    PhoSeedCollection;
  typedef edm::AssociationVector<reco::PhotonRefProd, std::vector<Frame1D> > PhoFrame1DCollection;
  typedef edm::AssociationVector<reco::PhotonRefProd, std::vector<Frame2D> > PhoFrame2DCollection;
  typedef edm::AssociationVector<reco::PhotonRefProd, std::vector<Frame3D> > PhoFrame3DCollection;
  typedef edm::AssociationVector<reco::PhotonRefProd, std::vector<Frame4D> > PhoFrame4DCollection;

}

#endif
