#ifndef RecoE2E_EGFrameProducer_h
#define RecoE2E_EGFrameProducer_h

#include <memory>
#include <iostream>
#include <vector>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h" // reco::PhotonCollection defined here
//#include "DataFormats/PatCandidates/interface/Photon.h"
//#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "RecoE2E/DataFormats/interface/FrameCollections.h"
#include "RecoE2E/FrameProducers/interface/FrameCropping.h"
#include "RecoE2E/FrameProducers/interface/constants.h"

using namespace std;

using reco::PhotonCollection;
using reco::PhotonRef;
//using pat::PhotonCollection;
//using pat::PhotonRef;

static const float ptCutEB    = 10.;        // min pt cut [GeV] for EB photons
static const float etaCutEB   = 1.44;       // max eta for EB photons
static const float defaultVal = -1.;        // default value to fill for invalid objects
static const unsigned int nSeedCoords = 2;  // no. of elements to specify frame seed coordinates
const unsigned int nFrameH = 32;            // frame height in no. of pixels
const unsigned int nFrameW = 32;            // frame width in no. of pixel
const unsigned int nDetEBenergyH = 170;     // detector image height in no. of pixels
const unsigned int nDetEBenergyW = 360;     // detector image width in no. of pixels

class EGFrameProducer : public edm::stream::EDProducer<> {

   public:

      explicit EGFrameProducer(const edm::ParameterSet&);
      ~EGFrameProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      // ----------member data ---------------------------
      // Input tokens
      edm::EDGetTokenT<EcalRecHitCollection> tEBRecHitCollection;
      edm::EDGetTokenT<PhotonCollection>     tPhotonCollection;
      edm::EDGetTokenT<e2e::Frame1D>         tEBenergyCollection;
      edm::EDGetTokenT<e2e::Frame3D>         tDetFramesCollection;
      edm::EDGetTokenT<e2e::Frame1D>         tChannelMappingCollection;

      // Handles
      edm::Handle<PhotonCollection>     hPhoton;
      edm::Handle<EcalRecHitCollection> hEBRecHits;
      edm::Handle<e2e::Frame1D>         hEBenergy;
      edm::Handle<e2e::Frame3D>         hDetFrames;
      edm::Handle<e2e::Frame1D>         hChannelMapping;

      e2e::Frame2D EBenergy_reshaped;
   
      // Detector image switches
      bool doEBenergy;

      // Output collections to be produced and values stored in them
      std::unique_ptr<e2e::PhoPredCollection>    cPhoProbs;
      std::unique_ptr<e2e::PhoSeedCollection>    cPhoSeeds;
      std::unique_ptr<e2e::Frame4D> cPhoFrames;
      std::unique_ptr<e2e::PhoFrame3DCollection> cPhoFrames3DCollection;
      std::vector<e2e::pred>    vPhoProbs;
      std::vector<e2e::seed>    vPhoSeeds;
      std::vector<e2e::Frame3D> vPhoFrames;

      // Frame dimensions determined at runtime
      int nPhos;   // frame batch size in no. of photons
      int nFrameD; // frame depth in no. of detector layers
      int ieta_global_offset_ = 55;

      // Seed finding
      void getEGseed ( e2e::seed&, const PhotonRef&, const edm::Handle<EcalRecHitCollection>& );
      float seedE;
      int ieta_, iphi_;

}; // EGFrameProducer

#endif
