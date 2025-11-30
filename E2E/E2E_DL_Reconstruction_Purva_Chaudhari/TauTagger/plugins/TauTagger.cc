//TauTagger
#include "RecoE2E/TauTagger/interface/TauTagger.h"

TauTagger::TauTagger(const edm::ParameterSet& iConfig)
{
  // Input tokens
  HBHERecHitCollectionT_  = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedHBHERecHitCollection"));
  jetCollectionT_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("ak8PFJetCollection"));
  genJetCollectionT_      = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("ak8GenJetCollection"));
  recoJetsT_              = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("ak8RecoJetsForBTagging"));
  JetFramesT_ = consumes<e2e::Frame4D>(iConfig.getParameter<edm::InputTag>("TauFrames"));
  siPixelRecHitCollectionT_ = consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("siPixelRecHitCollection"));
  siStripRecHitCollectionT_ = consumes<SiStripMatchedRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("siStripMatchedRecHitCollection"));
  siStripMatchedRecHitCollectionT_ = consumes<SiStripMatchedRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("siStripMatchedRecHitCollection")); 
  //tEGframeCollection = consumes<e2e::PhoFrame3DCollection>(iConfig.getParameter<edm::InputTag>("EGFrames"));

  mode_      = iConfig.getParameter<std::string>("mode");
  minJetPt_  = iConfig.getParameter<double>("minJetPt");
  maxJetEta_ = iConfig.getParameter<double>("maxJetEta");
  z0PVCut_   = iConfig.getParameter<double>("z0PVCut");
  
  // DL inference model
  modelName = iConfig.getParameter<std::string>("TauModelName");

  // Output collections to be produced
  //produces<e2e::PhoPredCollection>("TauProbs");
  produces<e2e::Frame2D>("TauProbs");
}

TauTagger::~TauTagger()
{
}

void
TauTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::LogInfo("TauTagger") << " >> Running TauTagger...";

  // Load required tokens into input collection handles
  iEvent.getByToken( jetCollectionT_, jets );
  iEvent.getByToken( JetFramesT_, hJetFrames );
  assert( hJetFrames->size() == jets->size() );
  
  e2e::Frame2D    vJetSeeds ( jets->size(), std::vector<float> (nSeedCoords, float(defaultVal)) );
  runEvtSel_jet ( iEvent, iSetup, vJetSeeds);
  assert( jets->size() == vJetSeeds.size() );
  
  nJets = jets->size();
  //std::vector<e2e::pred>    vJetProbs ( nJets, defaultVal );
  e2e::Frame2D vJetPred ( nJets );
  if (hJetFrames->size()>0) {
    // Get pointer to input Tau frames
    const std::vector<e2e::Frame3D>* pJetFrame = hJetFrames.product();
    nFrameD = pJetFrame->front().size(); // get size of depth dimension

    // Initialize product values to be stored with default values at start of every event
    // Each object is a vector over the no. of Jets in the event
  
    std::vector<e2e::Frame3D> vJetFrames( nJets,
                                        e2e::Frame3D(nFrameD,
                                        e2e::Frame2D(nFrameH,
                                        e2e::Frame1D(nFrameW, 0.))) );
    std::vector<e2e::Frame3D> vtmpFrames( nJets,
                                        e2e::Frame3D(1,
                                        e2e::Frame2D(nFrameH,
                                        e2e::Frame1D(nFrameW, 0.))) );

    //_____ Load Tau frame collection into `vJetFrames` for each jet _____//

    for ( unsigned int iJ = 0; iJ < vJetSeeds.size(); iJ++ ) {
      // Get Tau frame for this jet
      if (vJetSeeds[iJ][0]>=0 and vJetSeeds[iJ][1]>=0){
        vJetFrames[iJ] = pJetFrame->at(iJ);
        vtmpFrames[iJ][0] = vJetFrames[iJ][0];
      }
      else {
        //vJetFrames.push_back(e2e::Frame3D());
        //vtmpFrames.push_back(e2e::Frame3D());
      }
    } // jets

    //_____ Run DL inference _____//

    // Run inference on `vJetFrames` batch of size nJetss*nFrameD*nFrameH*nFrameW: store output in `vJetProbs`
    // Running on entire batch at once maximizes computing parellization
    // runInference( vJetProbs, vJetFrames, modelName );
    
    vJetPred = e2e::predict_tf(vJetFrames, modelName, "inputs","outputs");
    for (unsigned int iJ = 0; iJ < vJetSeeds.size(); iJ++){
      if ( vJetSeeds[iJ][0] == defaultVal and vJetSeeds[iJ][1] == defaultVal ){
        vJetPred[iJ] = e2e::Frame1D ({defaultVal});
      }
    }
  
    //_____ Store products associated with each photon _____//

    // Initialize pointers to edm::AssociationVector (key,val) collections
    // These collections create explicit associations between the photon object (key) and the stored product (val)
  }
  cJetProbs  = std::make_unique<e2e::Frame2D>   ( vJetPred );
    
  // Put collections into output EDM file
  iEvent.put( std::move(cJetProbs), "TauProbs" );

  return;
} // EGTagger::produce()

void
TauTagger::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
TauTagger::endStream()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TauTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TauTagger);
