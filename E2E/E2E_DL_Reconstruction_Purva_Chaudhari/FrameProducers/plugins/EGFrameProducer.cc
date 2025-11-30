#include "RecoE2E/FrameProducers/interface/EGFrameProducer.h"

EGFrameProducer::EGFrameProducer(const edm::ParameterSet& iConfig)
{
  // Input tokens
  tPhotonCollection   = consumes<PhotonCollection>(iConfig.getParameter<edm::InputTag>("photonCollection"));
  tEBenergyCollection = consumes<e2e::Frame1D>(iConfig.getParameter<edm::InputTag>("EBEnergy"));
  tEBRecHitCollection = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEBRecHitCollection"));
  tDetFramesCollection = consumes<e2e::Frame3D>(iConfig.getParameter<edm::InputTag>("DetFrames"));
  tChannelMappingCollection = consumes<e2e::Frame1D>(iConfig.getParameter<edm::InputTag>("ChannelMapping"));
  // Detector image switches
  doEBenergy = iConfig.getParameter<bool>("doEBenergy");

  // Output collections to be produced
  produces<e2e::PhoSeedCollection>   ("EGSeeds");
  produces<e2e::PhoFrame3DCollection> ("EGPhoFrames3DCollection");
  produces<e2e::Frame4D> ("EGFrames");
}

EGFrameProducer::~EGFrameProducer()
{
}

void
EGFrameProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo("EGFrameProducer") << " >> Running EGFrameProducer...";
  //std::cout<< " >> Running EGFrameProducer..."<<std::endl;
  // Load required tokens into input collection handles
  iEvent.getByToken( tPhotonCollection, hPhoton );
  if ( doEBenergy ) {
    //iEvent.getByToken( tEBenergyCollection, hEBenergy  );
    iEvent.getByToken( tChannelMappingCollection, hChannelMapping);
    iEvent.getByToken( tDetFramesCollection, hDetFrames);
    iEvent.getByToken( tEBRecHitCollection, hEBRecHits );
  }

  // Determine frame depth or no. of separate detector image channels
  nFrameD = 0;
  if ( doEBenergy ) nFrameD += 1;
  assert( nFrameD > 0 );

  // Initialize product values to be stored with default values at start of every event
  // Each object is a vector over the no. of photons in the event
  nPhos = hPhoton->size();
  std::vector<e2e::pred>    vPhoProbs ( nPhos, defaultVal );
  std::vector<e2e::seed>    vPhoSeeds ( nPhos, e2e::seed(nSeedCoords, int(defaultVal)) );
  std::vector<e2e::Frame3D> vPhoFrames( nPhos,
                                        e2e::Frame3D(nFrameD,
                                        e2e::Frame2D(nFrameH,
                                        e2e::Frame1D(nFrameW, 0.))) );
  
  edm::LogInfo("EGFrameProducer") << " >> Number of photon seeds: "<<nPhos;
  //_____ Find seed coordinates of each photon and crop frame around it  _____//
  //std::cout<<" >> Number of photon seeds: " << nPhos<<std::endl;
  for ( unsigned int iP = 0; iP < hPhoton->size(); iP++ ) {

    PhotonRef iRecoPho( hPhoton, iP );

    // Find photon seed: store output in `vPhoSeeds[iP]`
    getEGseed ( vPhoSeeds[iP], iRecoPho, hEBRecHits );
    edm::LogInfo("EGFrameProducer") << " >> seed(ieta,iphi):" << vPhoSeeds[iP][0] << "," << vPhoSeeds[iP][1];

    // Crop frame of size nFrameH*nFrameW for each detector image: store ouput in `vPhoFrames[iP]`
    // Each detector frame from iD=[0, nFrameD) should be stored in `vPhoFrames[iP][iD]`
    e2e::Frame3D vDetFrames = *hDetFrames;
    e2e::Frame1D vChannelMapping = *hChannelMapping;
    //std::cout<<"TTTPPP : "<<vDetFrames[1][0].size()<<std::endl;
    if (vChannelMapping[1] == 1){
      int ECALchannelIdx = vChannelMapping[0]+vChannelMapping[1] - 1;
      e2e::Frame2D EBenergy_reshaped (nDetEBenergyH, e2e::Frame1D(nDetEBenergyW, 0.));
      for (unsigned int x_idx=0; x_idx<nEBFrameH; x_idx++){
        for (unsigned int y_idx=0; y_idx<vDetFrames[ECALchannelIdx][0].size(); y_idx++){
          EBenergy_reshaped[x_idx][y_idx] = vDetFrames[ECALchannelIdx][x_idx+ieta_global_offset_][y_idx];
        }
      }
      if ( doEBenergy && vPhoSeeds[iP][0]>=0 && vPhoSeeds[iP][1]>=0) e2e::getFrame( vPhoFrames[iP][0], vPhoSeeds[iP], &EBenergy_reshaped/*hEBenergy.product()*/,
                                       2*EBDetId::MAX_IETA, EBDetId::MAX_IPHI );// can be removed if input is Frame2D
      edm::LogInfo("EGFrameProducer") << " >> vPhoFrames[iP][0][15][15]:" << vPhoFrames[iP][0][15][15];
    }
    else {
      //std::cout<<" >> ECAL stitched channel not present."<<std::endl; 
      edm::LogInfo("EGFrameProducer") << " >> ECAL stitched channel not present."; 
    }
  } // photons

  //_____ Store products associated with each photon _____//

  // Initialize pointers to edm::AssociationVector (key,val) collections
  // These collections create explicit associations between the photon object (key) and the stored product (val)
  cPhoSeeds  = std::make_unique<e2e::PhoSeedCollection>   ( reco::PhotonRefProd(hPhoton) );
  cPhoFrames3DCollection = std::make_unique<e2e::PhoFrame3DCollection>( reco::PhotonRefProd(hPhoton) );
  cPhoFrames = std::make_unique<e2e::Frame4D> (e2e::Frame4D(vPhoFrames));
  //std::unique_ptr<e2e::Frame4D> cPhoFrames (new e2e::Frame4D(vPhoFrames) );

  // Set association between photon ref (key) and products to be stored (val)
  for ( unsigned int iP = 0; iP < hPhoton->size(); iP++ ) {
    PhotonRef iRecoPho( hPhoton, iP );
    cPhoSeeds->setValue ( iP, vPhoSeeds[iP]  );
    cPhoFrames3DCollection->setValue( iP, vPhoFrames[iP] );
  } // photons

  // Put collections into output EDM file
  iEvent.put( std::move(cPhoSeeds),  "EGSeeds"  );
  iEvent.put( std::move(cPhoFrames3DCollection), "EGPhoFrames3DCollection" );
  iEvent.put( std::move(cPhoFrames), "EGFrames");

  return;

} // EGFrameProducer::produce()

//___________________________________________________________________________________//

/* Get EG seed coordinates:
 * Given a reconstructed photon object, determine detector image coordinates
 * corresponding to the position of its seed (i.e. crystal with largest energy deposit.
 * Store seed coordinates in `EGseed`.
 *
 * Inputs:
 *    EGseed:     ref to (empty) EG seed coordinates to be filled
 *    iRecoPho:   pointer to photon object
 *    hEBRecHits: pointer to EB detector rechit collection
*/
void EGFrameProducer::getEGseed ( e2e::seed& EGseed, const PhotonRef& iRecoPho, const edm::Handle<EcalRecHitCollection>& hEBRecHits )
{

  if ( iRecoPho->pt()       < ptCutEB  ) return;
  if ( abs(iRecoPho->eta()) > etaCutEB ) return; // TODO: implement EE seed finding

  // Get underlying super cluster (SC) or set of DetIds corresponding to energy deposits associated with photon
  reco::SuperClusterRef const& iSC = iRecoPho->superCluster();
  std::vector<std::pair<DetId, float>> const& SCHits( iSC->hitsAndFractions() );

  // Loop over SC hits of photon to find seed crystal with largest energy deposit
  seedE = defaultVal;
  iphi_ = int(defaultVal);
  ieta_ = int(defaultVal);
  for ( unsigned iH(0); iH != SCHits.size(); ++iH ) {

    // Get DetId
    if ( SCHits[iH].first.subdetId() != EcalBarrel ) continue;
    EcalRecHitCollection::const_iterator iRHit( hEBRecHits->find(SCHits[iH].first) );
    if ( iRHit == hEBRecHits->end() ) continue;

    // Convert coordinates to ordinals
    //EBDetId ebId( iSC->seed()->seed() ); // doesnt always seem to give largest deposit...
    EBDetId ebId( iRHit->id() );
    ieta_  = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta(); // [-85,...,-1,1,...,85]
    ieta_ += EBDetId::MAX_IETA; // [0,...,169]
    iphi_  = ebId.iphi()-1; // [0,...,359]

    // Store coordinates of seed
    if ( iRHit->energy() <= seedE ) continue;
    seedE     = float(iRHit->energy());
    EGseed[0] = ieta_;
    EGseed[1] = iphi_;

  } // SCHits

  edm::LogInfo("EGFrameProducer::getEGseed") << " >> photon seed(ieta,iphi,E):" << EGseed[0] << "," << EGseed[1] << "," << seedE;

  return;

} // EGFrameProducer::getEGseed()

void
EGFrameProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
EGFrameProducer::endStream()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EGFrameProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EGFrameProducer);
