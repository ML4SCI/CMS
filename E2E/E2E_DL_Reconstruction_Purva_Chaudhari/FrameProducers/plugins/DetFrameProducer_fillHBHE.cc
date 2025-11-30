#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

TH2F *hEvt_HBHE_energy;

// Fill HBHE rechits _________________________________________________________________//
void DetFrameProducer::fillHBHE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  // Intermediate helper histogram (single event only)
  hEvt_HBHE_energy = new TH2F("evt_HBHE_energy", "E(i#phi,i#eta);i#phi;i#eta",
      HBHE_IPHI_NUM,           HBHE_IPHI_MIN-1,    HBHE_IPHI_MAX,
      2*(HBHE_IETA_MAX_HE-1),-(HBHE_IETA_MAX_HE-1),HBHE_IETA_MAX_HE-1 );
  
  int iphi_, ieta_, ietaAbs_, idx_;
  float energy_;
  //float eta, GlobalPoint pos;

  vHBHE_energy_EB_.assign( 2*HBHE_IPHI_NUM*HBHE_IETA_MAX_EB, 0. );
  vHBHE_energy_.assign( 2*HBHE_IPHI_NUM*(HBHE_IETA_MAX_HE-1), 0. );
  hEvt_HBHE_energy->Reset();

  edm::Handle<HBHERecHitCollection> HBHERecHitsH_;
  iEvent.getByToken( HBHERecHitCollectionT_, HBHERecHitsH_ );

  /*
  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom;
  caloGeom = caloGeomH_.product();
  */

  // Fill HBHE rechits 
  for ( HBHERecHitCollection::const_iterator iRHit = HBHERecHitsH_->begin();
        iRHit != HBHERecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id and convert to histogram-friendly coordinates
    // NOTE: HBHE detector ids are indexed by (ieta,iphi,depth)!
    HcalDetId hId( iRHit->id() );
    // NOTE: HBHE iphi = 1 does not correspond to EB iphi = 1!
    // => Need to shift by 2 HBHE towers: HBHE::iphi: [1,...,71,72]->[3,4,...,71,72,1,2]
    iphi_  = hId.iphi() + 2; // shift
    iphi_  = iphi_ > HBHE_IPHI_MAX ? iphi_-HBHE_IPHI_MAX : iphi_; // wrap-around
    iphi_  = iphi_ - 1; // make histogram-friendly
    ietaAbs_  = hId.ietaAbs() == HBHE_IETA_MAX_HE ? HBHE_IETA_MAX_HE-1 : hId.ietaAbs(); // last HBHE ieta embedded
    ieta_  = hId.zside() > 0 ? ietaAbs_-1 : -ietaAbs_;
    //depth_ = hId.depth();

    // Fill vectors/histos by ieta range: /////////////////////////

    // (A) hId.ieta() > 20: full extent of HBHE
    // NOTE: HBHE iphis only occur in even numbers in coarse region
    // => Fill adjacent (odd) iphi bin and split energy evenly.
    if ( hId.ietaAbs() > HBHE_IETA_MAX_FINE ) {
      // Fill intermediate helper histogram
      hEvt_HBHE_energy->Fill( iphi_  , ieta_, energy_*0.5 );
      hEvt_HBHE_energy->Fill( iphi_+1, ieta_, energy_*0.5 );
      continue;
    }

    // (B) hId.ieta() <= 20: fine iphi granularity (beyond this, iphi granularity is halved)
    // Fill histograms normally
    else {
      hEvt_HBHE_energy->Fill( iphi_,ieta_,energy_ );
    }

    // (C) hId.ieta() <= 17: overlap with EB 
    // Additionally, fill EB-overlap-only vectors/histograms
    if ( hId.ietaAbs() > HBHE_IETA_MAX_EB ) continue;
    // Create hashed index: maps from [ieta][iphi][:] -> [idx_]
    // Effectively sums energies over depth for a given (ieta,iphi)
    idx_ = ( ieta_+HBHE_IETA_MAX_EB )*HBHE_IPHI_NUM + iphi_;
    // Fill vector for image
    // NOTE: Must use '+=' since different depths can map to same iphi,ieta
    vHBHE_energy_EB_[idx_] += energy_;
    //pos = caloGeom->getPosition( hId );
    //eta = pos.eta();
    //vHBHE_energy_EB_[idx_] += energy_/TMath::CosH(eta); // pick out transverse component only

  } // HBHE rechits

  // Fill vector for full HBHE image using helper histogram
  for (int ieta = 1; ieta < hEvt_HBHE_energy->GetNbinsY()+1; ieta++) {
    for (int iphi = 1; iphi < hEvt_HBHE_energy->GetNbinsX()+1; iphi++) {

      energy_ = hEvt_HBHE_energy->GetBinContent( iphi, ieta );
      if ( energy_ <= zs ) continue;
      idx_ = (ieta-1)*HBHE_IPHI_NUM + (iphi-1);
      // Fill vector for image
      vHBHE_energy_[idx_] = energy_;

    } // iphi
  } // ieta
}
