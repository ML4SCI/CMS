#ifndef RecoE2E_EGTagger_h
#define RecoE2E_EGTagger_h

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

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h" // reco::PhotonCollection defined here
//#include "DataFormats/PatCandidates/interface/Photon.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "RecoE2E/DataFormats/interface/FrameCollections.h"
#include "RecoE2E/FrameProducers/interface/EGFrameProducer.h"
#include "RecoE2E/FrameProducers/interface/predict_tf.h"

using namespace std;

using reco::PhotonCollection;
using reco::PhotonRef;
//using pat::PhotonCollection;
//using pat::PhotonRef;

class EGTagger : public edm::stream::EDProducer<> {

   public:

      explicit EGTagger(const edm::ParameterSet&);
      ~EGTagger();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      // ----------member data ---------------------------
      // Input tokens
      edm::EDGetTokenT<PhotonCollection>           tPhotonCollection;
      edm::EDGetTokenT<std::vector<e2e::Frame3D> > tEGframeCollection;
      //edm::EDGetTokenT<e2e::PhoFrame3DCollection> tEGframeCollection;
      // Handles
      edm::Handle<PhotonCollection>           hPhoton;
      edm::Handle<std::vector<e2e::Frame3D> > hEGframe;
      //edm::Handle<e2e::PhoFrame3DCollection> hEGframe;

      // DL inference model
      std::string modelName;
      void runInference( std::vector<e2e::pred>&, const std::vector<e2e::Frame3D>&, const std::string );

      // Vector to hold input EG frames for inference
      std::vector<e2e::Frame3D> vPhoFrames;

      // Frame dimensions determined at runtime
      int nPhos;   // frame batch size in no. of photons
      int nFrameD; // frame depth in no. of detector layers

      // Output collections to be produced and values stored in them
      std::unique_ptr<e2e::PhoPredCollection> cPhoProbs;
      std::vector<e2e::pred> vPhoProbs;

}; // EGTagger

#endif
