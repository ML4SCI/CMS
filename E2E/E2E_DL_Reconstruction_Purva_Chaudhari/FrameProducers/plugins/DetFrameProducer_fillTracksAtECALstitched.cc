#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

// All Tracks 
TH2F *hEvt_EE_tracksPt[nEE];
TH2F *hEvt_EE_tracksd0_PV[nEE];
TH2F *hEvt_EE_tracksdz_PV[nEE];
//std::vector<float> vECAL_tracksPt_;


// Function to map EE(phi,eta) histograms to ECAL(iphi,ieta) vector _______________________________//
void fillTracksAtECAL_with_EEproj (std::vector<float>& vECAL_tracksPt_, std::vector<float>& vECAL_tracksd0_PV_, std::vector<float>& vECAL_tracksdz_PV_, TH2F *hEvt_EE_tracksPt_, TH2F *hEvt_EE_tracksd0_PV, TH2F *hEvt_EE_tracksdz_PV, int ieta_global_offset ) {

  int ieta_global_;
  int ieta_, iphi_, idx_;
  float trackPt_, trackd0_PV_, trackdz_PV_;//, trackQPt_;
  
  for (int ieta = 1; ieta < hEvt_EE_tracksPt_->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    ieta_global_ = ieta_ + ieta_global_offset;
    for (int iphi = 1; iphi < hEvt_EE_tracksPt_->GetNbinsX()+1; iphi++) {

      trackPt_        = hEvt_EE_tracksPt_->GetBinContent( iphi, ieta );
      trackd0_PV_     = hEvt_EE_tracksd0_PV->GetBinContent( iphi, ieta );
      trackdz_PV_     = hEvt_EE_tracksdz_PV->GetBinContent( iphi, ieta );

      if ( (trackPt_ <= zs) ) continue;
      // NOTE: EB iphi = 1 does not correspond to physical phi = -pi so need to shift!
      iphi_ = iphi  + 5*38; // shift
      iphi_ = iphi_ > EB_IPHI_MAX ? iphi_-EB_IPHI_MAX : iphi_; // wrap-around
      iphi_ = iphi_ - 1;
      idx_  = ieta_global_*EB_IPHI_MAX + iphi_;
      // Fill vector for image
      vECAL_tracksPt_[idx_]  = trackPt_;
      vECAL_tracksd0_PV_[idx_]    = trackd0_PV_;
      vECAL_tracksdz_PV_[idx_]    = trackdz_PV_;
     } // iphi_
  } // ieta_

} // fillTracksAtECAL_with_EEproj

// Fill stitched EE-, EB, EE+ rechits ________________________________________________________//
void DetFrameProducer::fillTracksAtECALstitched ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  static const std::string strIndex[2] = {"m","p"};
  static const double* binIndex[2] = {eta_bins_EEm, eta_bins_EEp};
  for(unsigned int idx = 0; idx < 2; ++idx){
  hEvt_EE_tracksPt[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksPt").c_str(), "E(i#phi,i#eta);i#phi;i#eta",
				     EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
				     5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );
  hEvt_EE_tracksd0_PV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksd0_PV").c_str(), "d0(i#phi,i#eta);i#phi;i#eta",
					EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );
  hEvt_EE_tracksdz_PV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksdz_PV").c_str(), "dz(i#phi,i#eta);i#phi;i#eta",
					 EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					 5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );
  }
  
  int iphi_, ieta_, iz_, idx_;
  int ieta_global;
  int ieta_global_offset;
  float eta, phi, trackPt_, trackd0_, trackdz_;//trackQPt_, trackd0_, trackdz_, trackd0sig_, trackz0sig_;
  GlobalPoint pos;

  vECAL_tracksPt_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_tracksd0_PV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_tracksdz_PV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );

  for ( int iz(0); iz < nEE; ++iz ){
    hEvt_EE_tracksPt[iz]->Reset();
    hEvt_EE_tracksd0_PV[iz]->Reset();
    hEvt_EE_tracksdz_PV[iz]->Reset();
  }
  
  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_ );

  edm::Handle<EcalRecHitCollection> EERecHitsH_;
  iEvent.getByToken( EERecHitCollectionT_, EERecHitsH_ );

  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<reco::TrackCollection> tracksH_;
  iEvent.getByToken( trackCollectionT_, tracksH_ );

  edm::Handle<reco::VertexCollection> vertexInfo;
  iEvent.getByToken(vertexCollectionT_, vertexInfo);
  const reco::VertexCollection& vtxs = *vertexInfo;

  reco::Track::TrackQuality tkQt_ = reco::Track::qualityByName("highPurity");

  for ( reco::TrackCollection::const_iterator iTk = tracksH_->begin();
        iTk != tracksH_->end(); ++iTk ) {
    if ( !(iTk->quality(tkQt_)) ) continue;
    eta = iTk->eta();
    phi = iTk->phi();
    if ( std::abs(eta) > 3. ) continue;
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalBarrel ) continue;
    if ( id.subdetId() == EcalEndcap ) {
      iz_ = (eta > 0.) ? 1 : 0;
      // Fill intermediate helper histogram by eta,phi
      hEvt_EE_tracksPt[iz_]->Fill( phi, eta, iTk->pt() );
      const double z0 = ( !vtxs.empty() ? iTk->dz(vtxs[0].position()) : iTk->dz() );

      // if is PV
      if(fabs(z0) < z0PVCut_){
        const double d0 = ( !vtxs.empty() ? iTk->dxy(vtxs[0].position()) : iTk->dxy() );
        hEvt_EE_tracksd0_PV[iz_] ->Fill( phi, eta, d0 );
	hEvt_EE_tracksdz_PV[iz_] ->Fill( phi, eta, z0 );
      }
      else{ //if is not PV
        //hEvt_EE_tracksPt_nPV[iz_] ->Fill( phi, eta, iTk->pt() );
	//hEvt_EE_tracksQPt_nPV[iz_]->Fill( phi, eta, iTk->charge()*iTk->pt() );
      }
      
    }
  }  // tracks
  
  // Map EE-(phi,eta) to bottom part of ECAL(iphi,ieta)
  ieta_global_offset = 0;
  
  fillTracksAtECAL_with_EEproj( vECAL_tracksPt_, vECAL_tracksd0_PV_, vECAL_tracksdz_PV_, hEvt_EE_tracksPt[0], hEvt_EE_tracksd0_PV[0], hEvt_EE_tracksdz_PV[0], ieta_global_offset );
  
  // Fill middle part of ECAL(iphi,ieta) with the EB rechits.
  ieta_global_offset = 55;

  for ( reco::TrackCollection::const_iterator iTk = tracksH_->begin();
        iTk != tracksH_->end(); ++iTk ) { 

    eta = iTk->eta();
    phi = iTk->phi();
    trackPt_ = iTk->pt();
    trackd0_ =  ( !vtxs.empty() ? iTk->dxy(vtxs[0].position()) : iTk->dxy() );
    trackdz_ =  ( !vtxs.empty() ? iTk->dz(vtxs[0].position()) : iTk->dz() );
    if ( !(iTk->quality(tkQt_)) ) continue;

    if ( std::abs(eta) > 3. ) continue;
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalEndcap ) continue;
    if ( id.subdetId() == EcalBarrel ) { 
      EBDetId ebId( id );
      iphi_ = ebId.iphi() - 1;
      ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
      if ( trackPt_ <= zs ) continue;
      // Fill vector for image
      ieta_global = ieta_ + EB_IETA_MAX + ieta_global_offset;
      idx_ = ieta_global*EB_IPHI_MAX + iphi_; 
      vECAL_tracksPt_[idx_] += trackPt_;
      if(fabs(trackdz_) < z0PVCut_){
        vECAL_tracksd0_PV_[idx_] += trackd0_;
	vECAL_tracksdz_PV_[idx_] += trackdz_;
      }
      else{ //if is not PV
	//vECAL_tracksPt_nPV_[idx_] += trackPt_;
	//vECAL_tracksQPt_nPV_[idx_] += trackQPt_;
      }
    }
  } // EB Tracks
  
  // Map EE+(phi,eta) to upper part of ECAL(iphi,ieta)
  ieta_global_offset = ECAL_IETA_MAX_EXT + EB_IETA_MAX;
  
  fillTracksAtECAL_with_EEproj( vECAL_tracksPt_, vECAL_tracksd0_PV_, vECAL_tracksdz_PV_, hEvt_EE_tracksPt[1], hEvt_EE_tracksd0_PV[1], hEvt_EE_tracksdz_PV[1], ieta_global_offset );
} // fillTracksAtECALstitched()
