#include "RecoE2E/EGTagger/interface/EGTagger.h"

EGTagger::EGTagger(const edm::ParameterSet& iConfig)
{
  // Input tokens
  tPhotonCollection  = consumes<PhotonCollection>(iConfig.getParameter<edm::InputTag>("photonCollection"));
  tEGframeCollection = consumes<std::vector<e2e::Frame3D> >(iConfig.getParameter<edm::InputTag>("EGFrames"));
  //tEGframeCollection = consumes<e2e::PhoFrame3DCollection>(iConfig.getParameter<edm::InputTag>("EGFrames"));

  // DL inference model
  modelName = iConfig.getParameter<std::string>("EGModelName");

  // Output collections to be produced
  produces<e2e::PhoPredCollection>("EGProbs");
}

EGTagger::~EGTagger()
{
}

void
EGTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::LogInfo("EGTagger") << " >> Running EGTagger...";
  
  // Load required tokens into input collection handles
  iEvent.getByToken( tPhotonCollection,  hPhoton  );
  iEvent.getByToken( tEGframeCollection, hEGframe );
  assert( hPhoton->size() == hEGframe->size() );
  
  nPhos = hPhoton->size();
  std::vector<e2e::pred>    vPhoProbs ( nPhos, defaultVal );
  if (hEGframe->size()>0) {
    // Get pointer to input EG frames
    const std::vector<e2e::Frame3D>* pEGframe = hEGframe.product();
    nFrameD = pEGframe->front().size(); // get size of depth dimension

    // Initialize product values to be stored with default values at start of every event
    // Each object is a vector over the no. of photons in the event
  
    std::vector<e2e::Frame3D> vPhoFrames( nPhos,
                                        e2e::Frame3D(nFrameD,
                                        e2e::Frame2D(nFrameH,
                                        e2e::Frame1D(nFrameW, 0.))) );

    //_____ Load EG frame collection into `vPhoFrames` for each photon _____//
    
    for ( unsigned int iP = 0; iP < hPhoton->size(); iP++ ) {
      // Get EG frame for this photon
      PhotonRef iRecoPho( hPhoton, iP );
      vPhoFrames[iP] = pEGframe->at(iP);
    } // photons
     
    //_____ Run DL inference _____//

    // Run inference on `vPhoFrames` batch of size nPhos*nFrameD*nFrameH*nFrameW: store output in `vPhoProbs`
    // Running on entire batch at once maximizes computing parellization
    // runInference( vPhoProbs, vPhoFrames, modelName );
  
    e2e::Frame2D tmp_out = e2e::predict_tf(vPhoFrames, modelName, "inputs","outputs");
  
    //_____ Store products associated with each photon _____//

    // Initialize pointers to edm::AssociationVector (key,val) collections
    // These collections create explicit associations between the photon object (key) and the stored product (val)
  }
  cPhoProbs  = std::make_unique<e2e::PhoPredCollection>   ( reco::PhotonRefProd(hPhoton) );
  // Set association between photon ref (key) and products to be stored (val)
  for ( unsigned int iP = 0; iP < hPhoton->size(); iP++ ) {
    PhotonRef iRecoPho( hPhoton, iP );
    cPhoProbs->setValue( iP, vPhoProbs[iP] );
  } // photons
    
    
  // Put collections into output EDM file
  iEvent.put( std::move(cPhoProbs), "EGProbs" );

  return;
} // EGTagger::produce()

void
EGTagger::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
EGTagger::endStream()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EGTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EGTagger);
