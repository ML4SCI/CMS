#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

// Fill EB rechits _________________________________________________________________//
void DetFrameProducer::fillEB ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int idx_; // rows:ieta, cols:iphi
  float energy_;

  vEB_energy_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEB_time_.assign( EBDetId::kSizeForDenseIndexing, 0. );

  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_);

  // Fill EB rechits 
  for ( EcalRecHitCollection::const_iterator iRHit = EBRecHitsH_->begin();
        iRHit != EBRecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id and convert to histogram-friendly coordinates
    EBDetId ebId( iRHit->id() );
    
    // Get Hashed Index: provides convenient 
    // index mapping from [ieta][iphi] -> [idx]
    idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
    // Fill vectors for images
    vEB_energy_[idx_] = energy_;
    vEB_time_[idx_] = iRHit->time();

  } // EB rechits
 } // fillEB()
