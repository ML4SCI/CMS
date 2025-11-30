#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

// Fill TRK rec hits ////////////////////////////////
// by layer at ECAL stitched

TH2F *hBPIX_ECAL[nBPIX][Nhitproj];
//std::vector<float> vBPIX_ECAL_[nBPIX][Nhitproj];
TH2F *hEvt_EE_BPIX[nBPIX][nEE];

// Function to map EE(phi,eta) histograms to ECAL(iphi,ieta) vector _______________________________// 
void fillTRKLayerAtECAL_with_EEprojBPIX( TH2F *hEvt_EE_SUBDET, std::vector<float> & vSUBDET_ECAL_, int ieta_global_offset, int ieta_signed_offset ){
  int ieta_global_;// ieta_signed_;
  int ieta_, iphi_, idx_;
  float nEntries_=0.;
  for (int ieta = 1; ieta < hEvt_EE_SUBDET->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    ieta_global_ = ieta_ + ieta_global_offset;
    //ieta_signed_ = ieta_ + ieta_signed_offset;
    for (int iphi = 1; iphi < hEvt_EE_SUBDET->GetNbinsX()+1; iphi++) {
      nEntries_ = hEvt_EE_SUBDET->GetBinContent( iphi, ieta );
      if ( (nEntries_ == 0.) ) continue;
      // NOTE: EB iphi = 1 does not correspond to physical phi = -pi so need to shift!
      iphi_ = iphi  + 5*38; // shift
      iphi_ = iphi_ > EB_IPHI_MAX ? iphi_-EB_IPHI_MAX : iphi_; // wrap-around
      iphi_ = iphi_ - 1;
      idx_  = ieta_global_*EB_IPHI_MAX + iphi_;
      // Fill vector for image
      vSUBDET_ECAL_[idx_] = nEntries_;
      // Fill histogram for monitoring
    } // iphi_
  } // ieta_
} // fillTracksAtECAL_with_EEproj

void fillTRKLayerAtECAL_with_EEprojBPIX( TH2F *hEvt_EE_SUBDET[][nEE], std::vector<float> vSUBDET_ECAL_[][Nhitproj], int nSUBDET, unsigned int proj ){
  int ieta_global_offset,ieta_signed_offset;
  for(int nLayer=0; nLayer<nSUBDET; nLayer++){

    // Map EE-(phi,eta) to bottom part of ECAL(iphi,ieta)
    ieta_global_offset = 0;
    ieta_signed_offset = -ECAL_IETA_MAX_EXT;
    fillTRKLayerAtECAL_with_EEprojBPIX(hEvt_EE_SUBDET[nLayer][0], vSUBDET_ECAL_[nLayer][proj], ieta_global_offset, ieta_signed_offset);

    // Map EE+(phi,eta) to upper part of ECAL(iphi,ieta)
    ieta_global_offset = ECAL_IETA_MAX_EXT + EB_IETA_MAX;
    ieta_signed_offset = EB_IETA_MAX;
    fillTRKLayerAtECAL_with_EEprojBPIX(hEvt_EE_SUBDET[nLayer][1], vSUBDET_ECAL_[nLayer][proj], ieta_global_offset, ieta_signed_offset);
  }
}

void fillTRKLayerAtEBBPIX (DetId id, int layer_, unsigned int proj, std::vector<float> vSUBDET_ECAL_[][Nhitproj] ) {
  int ieta_global_offset = 55;
  EBDetId ebId( id );
  int iphi_ = ebId.iphi() - 1;
  int ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
  int ieta_signed = ieta_;
  int ieta_global = ieta_ + EB_IETA_MAX + ieta_global_offset;
  int idx_ = ieta_global*EB_IPHI_MAX + iphi_; 
  vSUBDET_ECAL_[layer_][proj][idx_] += 1.0;
}

void fillHelperAtEEBPIX ( float phi_, float eta_, int layer_, TH2F *hEvt_EE_SUBDET[][nEE]) {
  int iz_ = (eta_ > 0.) ? 1 : 0;
  hEvt_EE_SUBDET[layer_][iz_]->Fill( phi_, eta_);
}

unsigned int DetFrameProducer::getLayerBPIX(const DetId& detid, const TrackerTopology* tTopo) {

  unsigned int subid=detid.subdetId();
  switch(subid){

    case PixelSubdetector::PixelBarrel:{//BPIX
      PXBDetId pdetId = PXBDetId(detid);
      return pdetId.layer();
    }break;
  }
  return 999;
}

// Fill TRK rechits at ECAL stitched ______________________________________________________________//
void DetFrameProducer::fillTRKlayersAtECALstitchedBPIX ( const edm::Event& iEvent, const edm::EventSetup& iSetup, unsigned int proj ) {
  int layer=0;
  char hname[50], htitle[50];
  const double * eta_bins_EE[2] = {eta_bins_EEm,eta_bins_EEp};
  //BPIX branches
  for ( int iL(0); iL < nBPIX; iL++ ) {
      // Branches for images
      //layer = iL + 1;
      //sprintf(hname, "BPIX_layer%d_ECAL%s",layer,hit_projections[proj].c_str());
      //tree->Branch(hname,        &vBPIX_ECAL_[iL][proj]);
      // Histograms for monitoring
      //sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
      //hBPIX_ECAL[iL][proj] = fs->make<TH2F>(hname, htitle,
      //  EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      //  2*ECAL_IETA_MAX_EXT,-ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );
      if (proj==0){
        layer = iL + 1;
        for ( int iz(0); iz < nEE; iz++ ) {
          const char *zside = (iz > 0) ? "p" : "m";
          sprintf(hname, "evt_BPIX_layer%d_EE%s",layer,zside);
          sprintf(htitle,"N(ix,iy);ix;iy");
          hEvt_EE_BPIX[iL][iz] = new TH2F(hname, htitle,
          EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
          5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EE[iz] );
        } // iz
    }
  } // iL
  float eta, phi;
  GlobalPoint pos;

  for ( int iL(0); iL < nBPIX; iL++ ) {
    vBPIX_ECAL_[iL][proj].assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
    for ( int iz(0); iz < nEE; ++iz ) hEvt_EE_BPIX[iL][iz]->Reset();
  }
  //edm::Handle<TrackingRecHitCollection> TRKRecHitsH_;
  //iEvent.getByToken( TRKRecHitCollectionT_, TRKRecHitsH_ );
  //Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  bool isPVgood=false;
  edm::Handle<reco::VertexCollection> vertexInfo;
  iEvent.getByToken(vertexCollectionT_, vertexInfo);
  //const reco::VertexCollection& vtxs = *vertexInfo;
  isPVgood = vertexInfo.product()->size()>0;
  reco::Vertex the_PV;
  if (isPVgood) the_PV = vertexInfo.product()->at(0);
  TVector3 pv_v(the_PV.x(),the_PV.y(),the_PV.z());

  //sipixel
  edm::Handle<SiPixelRecHitCollection>  recHitColl;
  iEvent.getByToken(siPixelRecHitCollectionT_, recHitColl);

  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //std::cout <<" FOUND "<<(recHitColl.product())->dataSize()<<" Pixel Hits" << std::endl;

  SiPixelRecHitCollection::const_iterator recHitIdIterator    = (recHitColl.product())->begin();
  SiPixelRecHitCollection::const_iterator recHitIdIteratorEnd = (recHitColl.product())->end();

  for ( ; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++)
  {
    SiPixelRecHitCollection::DetSet detset = *recHitIdIterator;
    DetId detId = DetId(detset.detId()); // Get the Detid object
    //uint32_t TheID = detset.first;
    //DetId detId = DetId(TheID); // Get the Detid object
    unsigned int detType=detId.det(); // det type, tracker=1
    unsigned int subid=detId.subdetId(); //subdetector type, barrel=1, fpix=2
    //std::cout<<"Det: "<<detId.rawId()<<" "<<detId.null()<<" , type = "<<detType<<" , subId = "<<subid<<std::endl;

    if( detset.empty() ) {
      //std::cout << "detset is empty" << std::endl;
      edm::LogInfo("DetFrameProducer") << " !!! detset is empty !!! ";
      continue;
    }

    unsigned int layer = getLayerBPIX(detId, tTopo);
    const PixelGeomDetUnit* theGeomDet  = dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDetUnit(detId) );

    SiPixelRecHitCollection::DetSet::const_iterator pixeliter=detset.begin();
    SiPixelRecHitCollection::DetSet::const_iterator rechitRangeIteratorEnd   = detset.end();
  
    for(;pixeliter!=rechitRangeIteratorEnd;++pixeliter)
    {//loop on the rechit
      if (pixeliter->isValid())
      {
        LocalPoint lp = pixeliter->localPosition();
        GlobalPoint GP = theGeomDet->surface().toGlobal(Local3DPoint(lp));
        phi=0.;
        eta=0.;
        switch (proj)
        {
          case 0:
          {
            phi = GP.phi();
            eta = GP.eta();
            break;
          }
          case 1:
          {
            TVector3 GP_v(GP.x(),GP.y(),GP.z());
            GP_v=GP_v-pv_v;
            phi=GP_v.Phi();
            eta=GP_v.Eta();
            break;
          }
          default:
          {
            phi=0.;
            eta=0.;
            break;
          }
        }
        
        //if ( std::abs(eta) > 3. ) continue;
        DetId ecalId( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
        if ( subid == PixelSubdetector::PixelBarrel ){
          if ( ecalId.subdetId() == EcalBarrel ){
            fillTRKLayerAtEBBPIX ( ecalId, layer, proj, vBPIX_ECAL_ );
          }
          else if ( ecalId.subdetId() == EcalEndcap ){
            fillHelperAtEEBPIX ( phi, eta, layer, hEvt_EE_BPIX );
          }
        }
        else if ( subid == PixelSubdetector::PixelEndcap )
        {
          if ( ecalId.subdetId() == EcalBarrel ){
            //fillTRKLayerAtEB ( ecalId, layer, proj, hFPIX_ECAL, vFPIX_ECAL_ );
          }
          else if ( ecalId.subdetId() == EcalEndcap ){
            //fillHelperAtEE ( phi, eta, layer, hEvt_EE_FPIX);
          }
        }
      }
      else edm::LogInfo("DetFrameProducer") << " !!!!!!!!!!!!!! NO PIXEL HITS ARE VALID !!!!!!!!!!!!!!"; //std::cout << "!!!!!!!!!!!!!! NO PIXEL HITS ARE VALID !!!!!!!!!!!!!!" << std::endl;
    }
  }
  fillTRKLayerAtECAL_with_EEprojBPIX( hEvt_EE_BPIX, vBPIX_ECAL_, nBPIX, proj);
} // fillEB()
