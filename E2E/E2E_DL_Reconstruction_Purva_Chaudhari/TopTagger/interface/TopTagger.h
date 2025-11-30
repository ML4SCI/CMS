#ifndef RecoE2E_TopTagger_h
#define RecoE2E_TopTagger_h

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

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Calibration/IsolatedParticles/interface/DetIdFromEtaPhi.h"
#include "DQM/HcalCommon/interface/Constants.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h" // reco::PhotonCollection defined here
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TProfile2D.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TMath.h"
#include "TFile.h"
#include "TFrame.h"
#include "TBenchmark.h"
#include "TRandom.h"
#include "TSystem.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "RecoE2E/DataFormats/interface/FrameCollections.h"
#include "RecoE2E/FrameProducers/interface/JetFrameProducer.h"
#include "RecoE2E/FrameProducers/interface/predict_tf.h"

using namespace std;

//using pat::PhotonCollection;
//using pat::PhotonRef;

class TopTagger : public edm::stream::EDProducer<> {

   public:

      explicit TopTagger(const edm::ParameterSet&);
      ~TopTagger();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      // ----------member data ---------------------------
      // Input tokens
      edm::EDGetTokenT<reco::PFJetCollection> jetCollectionT_;
      edm::EDGetTokenT<e2e::Frame4D> JetFramesT_;
      edm::EDGetTokenT<reco::GenJetCollection> genJetCollectionT_;
      edm::EDGetTokenT<edm::View<reco::Jet> > recoJetsT_;
      edm::EDGetTokenT<HBHERecHitCollection> HBHERecHitCollectionT_;
      //edm::EDGetTokenT<e2e::PhoFrame3DCollection> tEGframeCollection;
      // Handles
      edm::Handle<reco::PFJetCollection> jets;
      edm::Handle<e2e::Frame4D> hJetFrames;
      //edm::Handle<e2e::PhoFrame3DCollection> hEGframe;
   
      // DL inference model
      std::string modelName;
      void runEvtSel_jet ( const edm::Event&, const edm::EventSetup&, e2e::Frame2D& );
      void runInference( std::vector<e2e::pred>&, const std::vector<e2e::Frame3D>&, const std::string );

      // Vector to hold input EG frames for inference
      std::vector<e2e::Frame3D> vJetFrames;

      // Frame dimensions determined at runtime
      int nJets;   // frame batch size in no. of photons
      int nFrameD; // frame depth in no. of detector layers
      std::string mode_;  // EventLevel / JetLevel
      double minJetPt_;
      double maxJetEta_;
      double z0PVCut_;

      // Output collections to be produced and values stored in them
      std::unique_ptr<e2e::Frame2D> cJetProbs;
      std::vector<e2e::pred> vJetProbs;

}; // TopTagger

#endif
