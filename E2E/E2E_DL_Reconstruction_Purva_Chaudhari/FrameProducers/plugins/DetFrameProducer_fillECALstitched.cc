#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

// Fill stitched EEm_EB_EEp image /////////////////////............/
// Store all ECAL event rechits into a stitched EEm_EB_EEp image 
// segmented in iphi,ieta spannning the full -3 < eta < 3. 
// Use EB-like granularity giving an extended range ieta=[-140,140].
//
// For endcaps, project EE hits into a helper histogram binned by 
// phi,eta before filling the full extended ECAL(iphi,eta) image.
// For barrel, fill EB hits directly since geometries are 1:1. 
//
// 'ieta_global' keeps track of the global ieta index count used
// for filling the extended image vector vECAL_energy.
// 'ieta_signed' keeps track of the position along [-140,140] used
// for filling the monitoring histogram hECAL_energy. (hECAL_energy is not written in the code below. It will have to be added if monitoring is needed.) 

TH2F *hEvt_EE_energy[nEE];

// Function to map EE(phi,eta) histograms to ECAL(iphi,ieta) vector _______________________________//
void fillECAL_with_EEproj ( std::vector<float>& vECAL_energy_, TH2F *hEvt_EE_energy_, int ieta_global_offset ) {
  
  int ieta_global_;
  int ieta_, iphi_, idx_;
  float energy_;

  for (int ieta = 1; ieta < hEvt_EE_energy_->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    ieta_global_ = ieta_ + ieta_global_offset;
    
    for (int iphi = 1; iphi < hEvt_EE_energy_->GetNbinsX()+1; iphi++) {

      energy_ = hEvt_EE_energy_->GetBinContent( iphi, ieta );
      if ( energy_ <= zs ) continue;
      // NOTE: EB iphi = 1 does not correspond to physical phi = -pi so need to shift!
      iphi_ = iphi  + 5*38; // shift
      iphi_ = iphi_ > EB_IPHI_MAX ? iphi_-EB_IPHI_MAX : iphi_; // wrap-around
      iphi_ = iphi_ - 1;
      idx_  = ieta_global_*EB_IPHI_MAX + iphi_;
      // Fill vector for image
      vECAL_energy_[idx_] = energy_;

    } // iphi_
  } // ieta_

} // fillECAL_with_EEproj

// Fill stitched EE-, EB, EE+ rechits ________________________________________________________//
void DetFrameProducer::fillECALstitched ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  // Intermediate helper histogram (single event only)
  hEvt_EE_energy[0] = new TH2F("evt_EEm_energy", "E(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
      5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EEm );
  hEvt_EE_energy[1] = new TH2F("evt_EEp_energy", "E(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
      5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EEp );
  
  int iphi_, ieta_, idx_;
  int ieta_global;
  int ieta_global_offset;
  float energy_;
  GlobalPoint pos;

  vECAL_energy_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  for ( int iz(0); iz < nEE; ++iz ) hEvt_EE_energy[iz]->Reset();

  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_ );
  edm::Handle<EcalRecHitCollection> EERecHitsH_;
  iEvent.getByToken( EERecHitCollectionT_, EERecHitsH_ );
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  // Fill the EE+/-(phi,eta) projection with the EE hits.
  for ( EcalRecHitCollection::const_iterator iRHit = EERecHitsH_->begin();
        iRHit != EERecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id
    EEDetId eeId( iRHit->id() );
    // Get position of cell centers
    pos  = caloGeom->getPosition( eeId );
    
  } // EE+/-

  // Map EE-(phi,eta) to bottom part of ECAL(iphi,ieta)
  ieta_global_offset = 0;
  
  fillECAL_with_EEproj( vECAL_energy_, hEvt_EE_energy[0], ieta_global_offset );

  // Fill middle part of ECAL(iphi,ieta) with the EB rechits.
  ieta_global_offset = 55;
  for ( EcalRecHitCollection::const_iterator iRHit = EBRecHitsH_->begin();
        iRHit != EBRecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id and convert to histogram-friendly coordinates
    EBDetId ebId( iRHit->id() );
    iphi_ = ebId.iphi() - 1;
    ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
    // Fill vector for image
    ieta_global = ieta_ + EB_IETA_MAX + ieta_global_offset;
    idx_ = ieta_global*EB_IPHI_MAX + iphi_; 
    vECAL_energy_[idx_] = energy_;

    pos  = caloGeom->getPosition( ebId );

  } // EB

  // Map EE+(phi,eta) to upper part of ECAL(iphi,ieta)
  ieta_global_offset = ECAL_IETA_MAX_EXT + EB_IETA_MAX;
  
  fillECAL_with_EEproj( vECAL_energy_, hEvt_EE_energy[1], ieta_global_offset);

} // fillECALstitched()
