#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

TH2F *hEvt_Adj_tracks[Nadjproj];
TH2F *hEvt_Adj_tracksPt[Nadjproj];
TH2F *hEvt_Adj_tracksPt_max[Nadjproj];

std::vector<int> DetFrameProducer::findSubcrystal(const CaloGeometry* caloGeom, const float& eta, const float& phi, const int& granularityMultiEta, const int& granularityMultiPhi)
{
    
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    std::vector<int> return_vector;
    if ( id.subdetId() == EcalBarrel )
    { 
      auto subDetGeometry = caloGeom->getSubdetectorGeometry(id);
      auto caloCellGeometry = subDetGeometry->getGeometry(id);
      auto cornersXYZ = caloCellGeometry->getCorners();
      std::vector<TVector3> corners;
      for (int i=0; i<5; i++)
      {
        TVector3 corner(cornersXYZ[i].x(),cornersXYZ[i].y(),cornersXYZ[i].z());
        corners.push_back(corner);
      }
      //float kappa= 4*TMath::Pi()/HBHE_IPHI_NUM;
      // if (phi>-kappa)
      //   phi=phi+kappa-TMath::Pi();
      // else  
      //   phi=phi+kappa+TMath::Pi();

      std::vector<float> eta_corners = {(float)corners[0].Eta(),(float)corners[1].Eta(),(float)corners[2].Eta(),(float)corners[3].Eta()};
      std::vector<float> phi_corners = {(float)corners[0].Phi(),(float)corners[1].Phi(),(float)corners[2].Phi(),(float)corners[3].Phi()};

      auto lowEta_lowPhi_index = 4;
      auto highEta_lowPhi_index = 4;
      auto lowEta_highPhi_index = 4;
      auto highEta_highPhi_index = 4;

      std::vector<size_t> eta_sorted_indices = {0,1,2,3};//(eta_corners.size());
      std::vector<size_t> phi_sorted_indices = {0,1,2,3};//(phi_corners.size());
      //std::iota(eta_sorted_indices.begin(), eta_sorted_indices.end(), 0);
      //std::iota(phi_sorted_indices.begin(), phi_sorted_indices.end(), 0);

      //index sort with lambdas
      std::sort(eta_sorted_indices.begin(), eta_sorted_indices.end(),[&eta_corners](size_t i1, size_t i2) {return eta_corners[i1] < eta_corners[i2];});
      std::sort(phi_sorted_indices.begin(), phi_sorted_indices.end(),[&phi_corners](size_t i1, size_t i2) {return phi_corners[i1] < phi_corners[i2];});

      for (unsigned int i =0; i<4; i++)
      {
        if      ((i==eta_sorted_indices[0] || i==eta_sorted_indices[1]) &&
                 (i==phi_sorted_indices[0] || i==phi_sorted_indices[1]) )
        { lowEta_lowPhi_index = i; }
        else if ((i==eta_sorted_indices[2] || i==eta_sorted_indices[3]) &&
                 (i==phi_sorted_indices[2] || i==phi_sorted_indices[3]) )
        { highEta_highPhi_index = i;  }
        else if ((i==eta_sorted_indices[0] || i==eta_sorted_indices[1]) &&
                 (i==phi_sorted_indices[2] || i==phi_sorted_indices[3]) )
        { lowEta_highPhi_index = i;  }
        else if ((i==eta_sorted_indices[2] || i==eta_sorted_indices[3]) &&
                 (i==phi_sorted_indices[0] || i==phi_sorted_indices[1]) )
        { highEta_lowPhi_index = i;  }
      }

      // if (lowEta_lowPhi_index  ==4 ||
      //     highEta_lowPhi_index ==4 ||
      //     lowEta_highPhi_index ==4 ||
      //     highEta_highPhi_index==4 ) std::cout<<"something went wrong\n";

      TVector2 lowEta_lowPhi_corner(corners[lowEta_lowPhi_index].Eta(),corners[lowEta_lowPhi_index].Phi());
      TVector2 highEta_lowPhi_corner(corners[highEta_lowPhi_index].Eta(),corners[highEta_lowPhi_index].Phi());
      TVector2 lowEta_highPhi_corner(corners[lowEta_highPhi_index].Eta(),corners[lowEta_highPhi_index].Phi());
      TVector2 highEta_highPhi_corner(corners[highEta_highPhi_index].Eta(),corners[highEta_highPhi_index].Phi());

      float subcrystal_eta_edges[granularityMultiEta+1][granularityMultiPhi+1];
      float subcrystal_phi_edges[granularityMultiEta+1][granularityMultiPhi+1];

      for (int etaIndex=0; etaIndex<granularityMultiEta+1; etaIndex++)
      {
        for (int phiIndex=0; phiIndex<granularityMultiPhi+1; phiIndex++)
        {
           TVector2 aveEta_lowPhi_corner (((granularityMultiEta-etaIndex)*lowEta_lowPhi_corner   + etaIndex*highEta_lowPhi_corner)/granularityMultiEta);
           TVector2 aveEta_highPhi_corner(((granularityMultiEta-etaIndex)*lowEta_highPhi_corner  + etaIndex*highEta_highPhi_corner)/granularityMultiEta);
           TVector2 aveEta_avePhi_corner (((granularityMultiPhi-phiIndex)*aveEta_lowPhi_corner   + phiIndex*aveEta_highPhi_corner )/granularityMultiPhi);
           subcrystal_eta_edges[etaIndex][phiIndex]=aveEta_avePhi_corner.X();
           subcrystal_phi_edges[etaIndex][phiIndex]=aveEta_avePhi_corner.Y();
        }
      }

      float subcrystal_eta_centers[granularityMultiEta][granularityMultiPhi];
      float subcrystal_phi_centers[granularityMultiEta][granularityMultiPhi];

      for (int etaIndex=0; etaIndex<granularityMultiEta; etaIndex++)
      {
        for (int phiIndex=0; phiIndex<granularityMultiPhi; phiIndex++)
        {
          float centerEta = (subcrystal_eta_edges[etaIndex][phiIndex]+
                             subcrystal_eta_edges[etaIndex+1][phiIndex+1]+
                             subcrystal_eta_edges[etaIndex+1][phiIndex]+
                             subcrystal_eta_edges[etaIndex][phiIndex+1])/4;

          float centerPhi = (subcrystal_phi_edges[etaIndex][phiIndex]+
                             subcrystal_phi_edges[etaIndex+1][phiIndex+1]+
                             subcrystal_phi_edges[etaIndex+1][phiIndex]+
                             subcrystal_phi_edges[etaIndex][phiIndex+1])/4;

          subcrystal_eta_centers[etaIndex][phiIndex]=centerEta;
          subcrystal_phi_centers[etaIndex][phiIndex]=centerPhi;          

        }
      }

      float minSubDist = 999;
      unsigned int subcrystal_phi_index=0;
      unsigned int subcrystal_eta_index=0;
      for (int etaIndex=0; etaIndex<granularityMultiEta; etaIndex++)
      {
        for (int phiIndex=0; phiIndex<granularityMultiPhi; phiIndex++)
        {
          float d=reco::deltaR(eta,phi,subcrystal_eta_centers[etaIndex][phiIndex],subcrystal_phi_centers[etaIndex][phiIndex]);
          if (d<minSubDist)
          {
            minSubDist = d;
            subcrystal_eta_index=etaIndex;
            subcrystal_phi_index=phiIndex;
          }
        }
      }

      //std::cout<<subcrystal_phi_index<<" "<<subcrystal_eta_index<<"\n";

      EBDetId ebId( id );
      // hEvt_Adj_tracksPt->SetBinContent(  ebId.iphi() - 1 +1, (ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta()) +141,
      //   hEvt_Adj_tracksPt->GetBinContent(ebId.iphi() - 1 +1, (ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta()) +141)+trackPt_ );
      int phi_base_coordinate = ebId.iphi() - 1 +1;
      int eta_base_coordinate =(ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta()) +141;
      phi_base_coordinate = (phi_base_coordinate - 1) * granularityMultiPhi + subcrystal_phi_index+1;
      eta_base_coordinate = (eta_base_coordinate - 1) * granularityMultiEta + subcrystal_eta_index+1;

      
      return_vector.push_back(phi_base_coordinate);
      return_vector.push_back(eta_base_coordinate);
    }
    return return_vector;
}

void DetFrameProducer::fillByBinNumber(TH2F * histo, const std::vector<int>& phi_eta, const float& value)
{
  histo->SetBinContent(phi_eta[0],phi_eta[1],
  histo->GetBinContent(phi_eta[0],phi_eta[1])+value);
}

// Fill adjustable EE-, EB, EE+ rechits ________________________________________________________//
void DetFrameProducer::fillTracksAtECALadjustable ( const edm::Event& iEvent, const edm::EventSetup& iSetup, unsigned int proj ) {
    
  for (unsigned int proj=0; proj<Nadjproj; proj++)
    {
        // Histograms for monitoring
        hEvt_Adj_tracks[proj] = new TH2F((std::string("evt_Adj_tracks")+adj_projections[proj]).c_str(), "E(#phi,#eta);#phi;#eta",
            totalPhiBins[proj], -TMath::Pi(), TMath::Pi(),
            adjEtaBins[proj].size()-1, &adjEtaBins[proj][0] );
        hEvt_Adj_tracksPt[proj] = new TH2F((std::string("evt_Adj_tracksPt")+adj_projections[proj]).c_str(), "E(#phi,#eta);#phi;#eta",
            totalPhiBins[proj], -TMath::Pi(), TMath::Pi(),
            adjEtaBins[proj].size()-1, &adjEtaBins[proj][0] );
        hEvt_Adj_tracksPt_max[proj] = new TH2F((std::string("evt_Adj_tracksPt_max")+adj_projections[proj]).c_str(), "E(#phi,#eta);#phi;#eta",
            totalPhiBins[proj], -TMath::Pi(), TMath::Pi(),
            adjEtaBins[proj].size()-1, &adjEtaBins[proj][0] );
    }
  //int iphi_, ieta_, iz_, idx_;
  //int ieta_global, ieta_signed;
  //int ieta_global_offset, ieta_signed_offset;
  float eta, phi, trackPt_; //trackQPt_,trackD0_, trackDz_;
  GlobalPoint pos;

  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  vECALadj_tracks_[proj].assign( totalEtaBins[proj]*totalPhiBins[proj], 0. );
  vECALadj_tracksPt_[proj].assign( totalEtaBins[proj]*totalPhiBins[proj], 0. );
  vECALadj_tracksPt_max_[proj].assign( totalEtaBins[proj]*totalPhiBins[proj], 0. );
  hEvt_Adj_tracks[proj]->Reset();
  hEvt_Adj_tracksPt[proj]->Reset();
  hEvt_Adj_tracksPt_max[proj]->Reset();
  
  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_ );
  edm::Handle<EcalRecHitCollection> EERecHitsH_;
  iEvent.getByToken( EERecHitCollectionT_, EERecHitsH_ );
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<reco::TrackCollection> tracksH_;
  iEvent.getByToken( trackCollectionT_, tracksH_ );

  
  bool isPVgood=false;
  edm::Handle<reco::VertexCollection> pvColl;
  iEvent.getByToken(pvCollectionT_, pvColl);
  isPVgood = pvColl.product()->size()>0;
  reco::Vertex the_PV;
  if (isPVgood) the_PV = pvColl.product()->at(0);

  reco::Track::TrackQuality tkQt_ = reco::Track::qualityByName("highPurity");

  int bin;

  for ( reco::TrackCollection::const_iterator iTk = tracksH_->begin();
        iTk != tracksH_->end(); ++iTk ) { 
    if ( !(iTk->quality(tkQt_)) ) continue;
    if(iTk->pt()<=0.5)continue; 
    if(iTk->charge()==0) continue;

    bool isPropagationOk=false;
    eta = 0.;
    phi = 0.;

    auto propagatedECALTrack = spr::propagateTrackToECAL(&*iTk, magfield.product());
    isPropagationOk=propagatedECALTrack.ok;
    if (propagatedECALTrack.ok)
    {
      eta = propagatedECALTrack.direction.eta();
      phi = propagatedECALTrack.direction.phi();
    }
    
    if ( std::abs(eta) > 3. || !isPropagationOk ) continue;

    trackPt_ = iTk->pt();
    
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalEndcap )
    {
      //int phi2;
      float kappa= 4*TMath::Pi()/HBHE_IPHI_NUM;
      if (phi>-kappa)
        phi=phi+kappa-TMath::Pi();
      else  
        phi=phi+kappa+TMath::Pi();

      hEvt_Adj_tracks[proj]->Fill( phi, eta );
      hEvt_Adj_tracksPt[proj]->Fill( phi, eta, trackPt_ );
      
      bin = hEvt_Adj_tracks[proj]->FindBin( phi, eta );
      if ( trackPt_ > hEvt_Adj_tracksPt_max[proj]->GetBinContent( bin ) ) {
        hEvt_Adj_tracksPt_max[proj]->SetBinContent( bin, trackPt_ );
      }
    }
    else if ( id.subdetId() == EcalBarrel ) { 

      std::vector<int> phi_eta = findSubcrystal(caloGeom, eta, phi, granularityMultiEta[proj], granularityMultiPhi[proj]);

      fillByBinNumber(hEvt_Adj_tracks[proj], phi_eta, 1.0);
      fillByBinNumber(hEvt_Adj_tracksPt[proj], phi_eta, trackPt_);
      if ( trackPt_ > hEvt_Adj_tracksPt_max[proj]->GetBinContent( phi_eta[0],phi_eta[1] ) ) {
        hEvt_Adj_tracksPt_max[proj]->SetBinContent( phi_eta[0],phi_eta[1], trackPt_ );
      }
    }
  } //tracks loop
  int index1d=0;
  for ( int ieta=1; ieta<=totalEtaBins[proj]; ieta++ )
  {
    for ( int iphi=1; iphi<=totalPhiBins[proj]; iphi++ )
    {
      index1d= (ieta-1)*totalPhiBins[proj]+iphi-1;//ieta_global*EB_IPHI_MAX + iphi_; 
      vECALadj_tracksPt_[proj][index1d]=hEvt_Adj_tracksPt[proj]->GetBinContent(iphi,ieta);
      vECALadj_tracks_[proj][index1d]=hEvt_Adj_tracks[proj]->GetBinContent(iphi,ieta);
      vECALadj_tracksPt_max_[proj][index1d]=hEvt_Adj_tracksPt_max[proj]->GetBinContent(iphi,ieta);
    }
  }
} // fillTracksAtECALadjustable()
