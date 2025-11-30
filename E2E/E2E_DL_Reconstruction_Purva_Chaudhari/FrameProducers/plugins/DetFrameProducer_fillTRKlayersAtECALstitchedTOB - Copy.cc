#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"

// Fill TRK rec hits ////////////////////////////////
// by layer at ECAL stitched

TH2F *hTOB_ECAL[nTOB][Nhitproj];
//std::vector<float> vTOB_ECAL_[nTOB][Nhitproj];
TH2F *hEvt_EE_TOB[nTOB][nEE];

TH2F *hTEC_ECAL[nTEC][Nhitproj];
//std::vector<float> vTEC_ECAL_[nTEC][Nhitproj];
TH2F *hEvt_EE_TEC[nTEC][nEE];

TH2F *hTIB_ECAL[nTIB][Nhitproj];
//std::vector<float> vTIB_ECAL_[nTIB][Nhitproj];
TH2F *hEvt_EE_TIB[nTIB][nEE];

TH2F *hTID_ECAL[nTID][Nhitproj];
//std::vector<float> vTID_ECAL_[nTID][Nhitproj];
TH2F *hEvt_EE_TID[nTID][nEE];

// Function to map EE(phi,eta) histograms to ECAL(iphi,ieta) vector _______________________________// 
void fillTRKLayerAtECAL_with_EEproj( TH2F *hEvt_EE_SUBDET, std::vector<float> & vSUBDET_ECAL_, TH2F *hSUBDET_ECAL, int ieta_global_offset, int ieta_signed_offset ){
  int ieta_global_, ieta_signed_;
  int ieta_, iphi_, idx_;
  float nEntries_=0.;
  for (int ieta = 1; ieta < hEvt_EE_SUBDET->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    ieta_global_ = ieta_ + ieta_global_offset;
//    ieta_signed_ = ieta_ + ieta_signed_offset;
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
 //     hSUBDET_ECAL->Fill( iphi_, ieta_signed_, nEntries_ );
    } // iphi_
  } // ieta_
} // fillTracksAtECAL_with_EEproj

void fillTRKLayerAtECAL_with_EEproj( TH2F *hEvt_EE_SUBDET[][nEE], std::vector<float> vSUBDET_ECAL_[][Nhitproj], TH2F *hSUBDET_ECAL[][Nhitproj], int nSUBDET, unsigned int proj ){
  int ieta_global_offset,ieta_signed_offset;
  for(int nLayer=0; nLayer<nSUBDET; nLayer++){

    // Map EE-(phi,eta) to bottom part of ECAL(iphi,ieta)
    ieta_global_offset = 0;
    ieta_signed_offset = -ECAL_IETA_MAX_EXT;
    fillTRKLayerAtECAL_with_EEproj(hEvt_EE_SUBDET[nLayer][0], vSUBDET_ECAL_[nLayer][proj], hSUBDET_ECAL[nLayer][proj], ieta_global_offset, ieta_signed_offset);

    // Map EE+(phi,eta) to upper part of ECAL(iphi,ieta)
    ieta_global_offset = ECAL_IETA_MAX_EXT + EB_IETA_MAX;
    ieta_signed_offset = EB_IETA_MAX;
    fillTRKLayerAtECAL_with_EEproj(hEvt_EE_SUBDET[nLayer][1], vSUBDET_ECAL_[nLayer][proj], hSUBDET_ECAL[nLayer][proj], ieta_global_offset, ieta_signed_offset);
  }
}

void fillTRKLayerAtEB (DetId id, int layer_, unsigned int proj, TH2F *hSUBDET_ECAL[][Nhitproj], std::vector<float> vSUBDET_ECAL_[][Nhitproj] ) {
  int ieta_global_offset = 55;
  EBDetId ebId( id );
  int iphi_ = ebId.iphi() - 1;
  int ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
 // int ieta_signed = ieta_;
  int ieta_global = ieta_ + EB_IETA_MAX + ieta_global_offset;
  int idx_ = ieta_global*EB_IPHI_MAX + iphi_; 
  vSUBDET_ECAL_[layer_-1][proj][idx_] += 1.0;
 // hSUBDET_ECAL[layer_-1][proj]->Fill( iphi_, ieta_signed, 1. );
}

void fillHelperAtEE ( float phi_, float eta_, int layer_, TH2F *hEvt_EE_SUBDET[][nEE]) {
  int iz_ = (eta_ > 0.) ? 1 : 0;
 // hEvt_EE_SUBDET[layer_-1][iz_]->Fill( phi_, eta_);
}

unsigned int DetFrameProducer::getLayer(const DetId& detid, const TrackerTopology* tTopo) {
 //                                            uint16_t
 // +------------+---------------+---------------------------+-----------------+----------------+
 // |  tk/mu/mtd | sub-structure |     sub-sub-structure     |     stereo      |    hit type    |
 // +------------+---------------+---------------------------+-----------------+----------------+
 // |    11-10   | 9   8    7    |  6     5     4     3      |        2        |    1        0  |  bit
 // +------------+---------------+---------------------------+-----------------+----------------|
 // | tk  = 1    |    PXB = 1    | layer = 1-3               |                 | hit type = 0-3 |
 // | tk  = 1    |    PXF = 2    | disk  = 1-2               |                 | hit type = 0-3 |
 // | tk  = 1    |    TIB = 3    | layer = 1-4               | 0=rphi,1=stereo | hit type = 0-3 |
 // | tk  = 1    |    TID = 4    | wheel = 1-3               | 0=rphi,1=stereo | hit type = 0-3 |
 // | tk  = 1    |    TOB = 5    | layer = 1-6               | 0=rphi,1=stereo | hit type = 0-3 |
 // | tk  = 1    |    TEC = 6    | wheel = 1-9               | 0=rphi,1=stereo | hit type = 0-3 |
 // | mu  = 0    |    DT  = 1    | 4*(stat-1)+superlayer     |                 | hit type = 0-3 |
 // | mu  = 0    |    CSC = 2    | 4*(stat-1)+(ring-1)       |                 | hit type = 0-3 |
 // | mu  = 0    |    RPC = 3    | 4*(stat-1)+2*layer+region |                 | hit type = 0-3 |
 // | mu  = 0    |    GEM = 4    | 2*(stat-1)+2*(layer-1)    |                 | hit type = 0-3 |
 // | mu  = 0    |    ME0 = 5    | roll                      |                 | hit type = 0-3 |
 // | mtd = 2    |    BTL = 1    | moduleType = 1-3          |                 | hit type = 0-3 | 
 // | mtd = 2    |    ETL = 2    | ring = 1-12               |                 | hit type = 0-3 | 
 // +------------+---------------+---------------------------+-----------------+----------------+
  unsigned int  subid=detid.subdetId();
  switch(subid){

    case SiStripDetId::TIB:{//TIB
      SiStripDetId pdetId = SiStripDetId(detid.rawId());
      return tTopo->tibLayer(pdetId);
      //return tTopo->layer(SiStripDetId(detid));
      //return tTopo->tibLayer(detid.rawId());
      //return tTopo->tibLayer(detid.rawId());
    }break;

    case SiStripDetId::TID:{//TID
      SiStripDetId pdetId = SiStripDetId(detid.rawId());
      return tTopo->tidWheel(pdetId);
      //return tTopo->layer(SiStripDetId(detid));
      //return tTopo->tidWheel(SiStripDetId(detid));
    }break;

    case SiStripDetId::TOB:{//TOB
      SiStripDetId pdetId = SiStripDetId(detid.rawId());
      return tTopo->tobLayer(pdetId);
      //return tTopo->layer(SiStripDetId(detid));
      //return tTopo->tobLayer(detid.rawId());
    }break;

    case SiStripDetId::TEC:{//TEC
      SiStripDetId pdetId = SiStripDetId(detid.rawId());
      return  tTopo->tecWheel(pdetId);
      //return tTopo->layer(SiStripDetId(detid));
      //return tTopo->tecWheel(detid.rawId());
    }break;

  }
  return 999;
}

// Fill TRK rechits at ECAL stitched ______________________________________________________________//
void DetFrameProducer::fillTRKlayersAtECALstitched ( const edm::Event& iEvent, const edm::EventSetup& iSetup, unsigned int proj ) {
  
  //conf_=iConfig;
  int layer=0;
  char hname[50], htitle[50];
  const double * eta_bins_EE[2] = {eta_bins_EEm,eta_bins_EEp};
  //TOB 
    for ( int iL(0); iL < nTOB; iL++ ) {
    // Branches for images
    //layer = iL + 1;
    //sprintf(hname, "TOB_layer%d_ECAL%s",layer,hit_projections[proj].c_str());
    //tree->Branch(hname,        &vTOB_ECAL_[iL][proj]);
    // Histograms for monitoring
    //sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    //TOB_ECAL[iL][proj] = fs->make<TH2F>(hname, htitle,
    //  EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
    //  2*ECAL_IETA_MAX_EXT,-ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );
    if (proj==0){
	layer = iL + 1;
      for ( int iz(0); iz < nEE; iz++ ) {
      	const char *zside = (iz > 0) ? "p" : "m";
      	sprintf(hname, "evt_TOB_layer%d_EE%s",layer,zside);
      	sprintf(htitle,"N(ix,iy);ix;iy");
      	hEvt_EE_TOB[iL][iz] = new TH2F(hname, htitle,
      	EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
      	5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EE[iz] );
      } // iz
     }
    } // iL
  

    //TEC
    for ( int iL(0); iL < nTEC; iL++ ) {
      // Branches for images
      //layer = iL + 1;
      //sprintf(hname, "TEC_layer%d_ECAL%s",layer,hit_projections[proj].c_str());
      //tree->Branch(hname,        &vTEC_ECAL_[iL][proj]);
      // Histograms for monitoring
      //sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
      //hTEC_ECAL[iL][proj] = fs->make<TH2F>(hname, htitle,
      // EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      // 2*ECAL_IETA_MAX_EXT,-ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );
      if (proj==0){
	layer = iL + 1;
        for ( int iz(0); iz < nEE; iz++ ) {
          const char *zside = (iz > 0) ? "p" : "m";
          sprintf(hname, "evt_TEC_layer%d_EE%s",layer,zside);
          sprintf(htitle,"N(ix,iy);ix;iy");
          hEvt_EE_TEC[iL][iz] = new TH2F(hname, htitle,
          EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
          5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EE[iz] );
        } // iz
      }
    } // iL

    //TIB
    for ( int iL(0); iL < nTIB; iL++ ) {
      // Branches for images
      //layer = iL + 1;
      //sprintf(hname, "TIB_layer%d_ECAL%s",layer,hit_projections[proj].c_str());
      //tree->Branch(hname,        &vTIB_ECAL_[iL][proj]);
      // Histograms for monitoring
      //sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
      //hTIB_ECAL[iL][proj] = fs->make<TH2F>(hname, htitle,
      //  EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      //  2*ECAL_IETA_MAX_EXT,-ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );
      if (proj==0){
        layer = iL + 1;
	for ( int iz(0); iz < nEE; iz++ ) {
          const char *zside = (iz > 0) ? "p" : "m";
          sprintf(hname, "evt_TIB_layer%d_EE%s",layer,zside);
          sprintf(htitle,"N(ix,iy);ix;iy");
          hEvt_EE_TIB[iL][iz] = new TH2F(hname, htitle,
          EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
          5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EE[iz] );
        } // iz
      }
    } // iL

    //TID
    for ( int iL(0); iL < nTID; iL++ ) {
      // Branches for images
      // layer = iL + 1;
      //sprintf(hname, "TID_layer%d_ECAL%s",layer,hit_projections[proj].c_str());
      //tree->Branch(hname,        &vTID_ECAL_[iL][proj]);
      // Histograms for monitoring
      //sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
      //hTID_ECAL[iL][proj] = fs->make<TH2F>(hname, htitle,
      //  EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      //  2*ECAL_IETA_MAX_EXT,-ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );
      if (proj==0){
	layer = iL + 1;	
        for ( int iz(0); iz < nEE; iz++ ) {
          const char *zside = (iz > 0) ? "p" : "m";
          sprintf(hname, "evt_TID_layer%d_EE%s",layer,zside);
          sprintf(htitle,"N(ix,iy);ix;iy");
          hEvt_EE_TID[iL][iz] = new TH2F(hname, htitle,
          EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
          5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), eta_bins_EE[iz] );
        } // iz
     }
  } // iL

  float eta, phi;
  GlobalPoint pos;

  for ( int iL(0); iL < nTOB; iL++ ) {
    vTOB_ECAL_[iL][proj].assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
    for ( int iz(0); iz < nEE; ++iz ) hEvt_EE_TOB[iL][iz]->Reset();
  }
  for ( int iL(0); iL < nTEC; iL++ ) {
    vTEC_ECAL_[iL][proj].assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
    for ( int iz(0); iz < nEE; ++iz ) hEvt_EE_TEC[iL][iz]->Reset();
  }
  for ( int iL(0); iL < nTIB; iL++ ) {
    vTIB_ECAL_[iL][proj].assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
    for ( int iz(0); iz < nEE; ++iz ) hEvt_EE_TIB[iL][iz]->Reset();
  }
  for ( int iL(0); iL < nTID; iL++ ) {
    vTID_ECAL_[iL][proj].assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
    for ( int iz(0); iz < nEE; ++iz ) hEvt_EE_TID[iL][iz]->Reset();
  }

  //edm::Handle<TrackingRecHitCollection> TRKRecHitsH_;
  //iEvent.getByToken( TRKRecHitCollectionT_, TRKRecHitsH_ );
  // Provides access to global cell position
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

  edm::Handle<SiStripMatchedRecHit2DCollection>  stripRecHitColl;
  iEvent.getByToken(siStripRecHitCollectionT_, stripRecHitColl);

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
    if( detset.empty() ) {
      edm::LogInfo("DetFrameProducer") << " !!! detset is empty !!! ";
      continue;
    }
    unsigned int layer = getLayer(detId, tTopo);
    //std::cout<<"Pixel Id = : "<<detId.rawId()<<" "<<detId.null()<<" , type = "<<detType<<" - subId ( 1->bpix | 2->fpix ) = "<< subid << " - Layer = " << layer << std::endl;
    const PixelGeomDetUnit* theGeomDet  = dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDetUnit(detId) );
     
  }
  // --  siSTRIP --
  
  for ( SiStripMatchedRecHit2DCollection::const_iterator detunit_iterator = stripRecHitColl->begin(), detunit_end = stripRecHitColl->end(); detunit_iterator != detunit_end; ++detunit_iterator) {
    SiStripMatchedRecHit2DCollection::DetSet rechitRange = *detunit_iterator;
    DetId detId = DetId(detunit_iterator->detId());
    unsigned int id = detunit_iterator->detId();
    unsigned int subid=detId.subdetId();
    unsigned int layer = getLayer(id, tTopo);
//    std::cout << "Strip Id = " << id << " - subId ( 3->TIB | 4->TID | 5->TOB | 6->TEC ) = " << subid << " - Layer = " << layer <<  std::endl;
    const StripGeomDetUnit* stripDet = (const StripGeomDetUnit*)theTracker.idToDet(detId);
    if(stripDet==0) {
     // std::cout << "SiStripRecHitConverter: Detid=" << id << " not found, trying next one" << std::endl;
      continue;
    }
    const StripTopology * stripTopol = (StripTopology*)(&stripDet->topology()); ;
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = rechitRange.begin();
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd   = rechitRange.end();
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator stripiter=rechitRangeIteratorBegin;
    unsigned int iRecHit = 0;
    for(stripiter=rechitRangeIteratorBegin;stripiter!=rechitRangeIteratorEnd;++stripiter){//loop on the rechit
      if (stripiter->isValid()){
        iRecHit++;
        SiStripMatchedRecHit2D const rechit=*stripiter;
        const GeomDet* stripdet=rechit.det();
        //DetId stripid=rechit.geographicalId();
        //std::vector<const SiStripCluster*> clust=rechit.cluster();
        LocalPoint lp = rechit.localPosition();
        GlobalPoint GP = stripDet->surface().toGlobal(Local3DPoint(lp));
     //   std::cout << " " << iRecHit << " | global position: x = " << GP.x() << " , y = "<< GP.y() << " , z = " << GP.z() <<std::endl;
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
	//std::cout<<"Break 1\n";
	//std::cout<<"Subid : "<<subid<<" Type "<<typeid(subid).name()<<"\n";       
	//std::cout<<"Strip Subdetector TOB: "<<StripSubdetector::TOB <<"\n";
	//std::cout<<"Strip Subdetector TIB : "<<StripSubdetector::TIB <<" Type "<<typeid(StripSubdetector::TIB).name()<<"\n";
	//std::cout<<"Strip Subdetector TEC: "<<StripSubdetector::TEC <<"\n";
	//std::cout<<"Strip Subdetector TID: "<<StripSubdetector::TID <<"\n";
//if ( std::abs(eta) > 3. ) continue;
        DetId ecalId( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
        //std::cout<<"Subid : "<<subid<<" Type "<<typeid(subid).name()<<"\n";
        //std::cout<<"Strip Subdetector TOB: "<<StripSubdetector::TOB <<"\n";
        //std::cout<<"Strip Subdetector TIB : "<<StripSubdetector::TIB <<" Type "<<typeid(StripSubdetector::TIB).name()<<"\n";
       //std::cout<<"Stage1: "<<subid<<std::endl;
     
 //      std::cout<<"Layer: "<<layer<<" iphi : "<<iphi_<<" ieta : "<<ieta_<<" ieta global : "<<ieta_global<<" idx : "<<idx_<<std::endl;
       //std::cout<<sizeof(vTOB_ECAL_)/sizeof(vTOB_ECAL_[0])<<"\n";
       //std::cout<<sizeof(vTOB_ECAL_[0])/sizeof(vTOB_ECAL_[0][0])<<"\n";
       //std::cout<<sizeof(vTOB_ECAL_[0][0])/sizeof(vTOB_ECAL_[0][0][0])<<"\n"; 
       //std::cout<<sizeof(vTOB_ECAL_[0][0][0])/sizeof(vTOB_ECAL_[0][0][0][0])<<"\n";
       //std::cout<<sizeof(vTID_ECAL_)/sizeof(vTID_ECAL_[0])<<"\n";
       //std::cout<<sizeof(vTID_ECAL_[0])/sizeof(vTID_ECAL_[0][0])<<"\n";
       //std::cout<<sizeof(vTID_ECAL_[0][0])/sizeof(vTID_ECAL_[0][0][0])<<"\n";
      
       if ( subid == StripSubdetector::TOB ) {
	//std::cout<<"STAGE 1";
          if ( ecalId.subdetId() == EcalBarrel ){
		fillTRKLayerAtEB ( ecalId, layer, proj, hTOB_ECAL, vTOB_ECAL_ );
		//std::cout<<"Break 2A";
          }
          else if ( ecalId.subdetId() == EcalEndcap ){
            fillHelperAtEE ( phi, eta, layer, hEvt_EE_TOB);
		//std::cout<<"Break 2B";
          }
        }
        //std::cout<<"Stage2: "<<subid<<std::endl;
	else if ( subid == StripSubdetector::TIB ) {
	   // std::cout<<"STAGE 2";
            //std::cout<<"ECAL id"<<ecalId.subdetId()<<" Ecal barrel "<<EcalBarrel<<" Ecal Endcap"<<EcalEndcap;
//	    vTIB_ECAL_[layer-1][proj][idx_] += 1.0;
	   if ( ecalId.subdetId() == EcalBarrel ){
            fillTRKLayerAtEB ( ecalId, layer, proj, hTIB_ECAL, vTIB_ECAL_ );
            //std::cout<<"Break 3A";  
	    }
            else if ( ecalId.subdetId() == EcalEndcap ){
               fillHelperAtEE ( phi, eta, layer, hEvt_EE_TIB);
		//std::cout<<"Break 3B";
            }
        }
        //std::cout<<"Stage3: "<<subid<<std::endl;
	if ( subid == StripSubdetector::TEC) {
		//std::cout<<"STAGE 3";
          if ( ecalId.subdetId() == EcalBarrel ){
            fillTRKLayerAtEB ( ecalId, layer, proj, hTEC_ECAL, vTEC_ECAL_ );
		//std::cout<<"Break 4A";
          }
          else if ( ecalId.subdetId() == EcalEndcap ){
            fillHelperAtEE ( phi, eta, layer, hEvt_EE_TEC);
		//std::cout<<"Break 4B";
          }
        }
        else if ( subid == StripSubdetector::TID ) {
	//	std::cout<<"STAGE 4";
          if ( ecalId.subdetId() == EcalBarrel ){

            fillTRKLayerAtEB ( ecalId, layer, proj, hTID_ECAL, vTID_ECAL_ );
         //	std::cout<<"Break 5A"; 
	 }
          else if ( ecalId.subdetId() == EcalEndcap ){
            fillHelperAtEE ( phi, eta, layer, hEvt_EE_TID);
         //	std::cout<<"Break 5B"; 
	 }
        }
	//std::cout<<"STAGE 5";
      } else edm::LogInfo("DetFrameProducer") << " !!!!!!!!!!!!!! NO STRIP HITS ARE VALID !!!!!!!!!!!!!!";// std::cout << "!!!!!!!!!!!!!! NO STRIP HITS ARE VALID !!!!!!!!!!!!!!" << std::endl;
    }// std::cout << "End of StripReCHit " << iRecHit << std::endl;
  } // end loop over detectors 
  
  fillTRKLayerAtECAL_with_EEproj( hEvt_EE_TOB, vTOB_ECAL_, hTOB_ECAL, nTOB, proj);
  fillTRKLayerAtECAL_with_EEproj( hEvt_EE_TIB, vTIB_ECAL_, hTIB_ECAL, nTIB, proj);
  fillTRKLayerAtECAL_with_EEproj( hEvt_EE_TEC, vTEC_ECAL_, hTEC_ECAL, nTEC, proj);
  fillTRKLayerAtECAL_with_EEproj( hEvt_EE_TID, vTID_ECAL_, hTID_ECAL, nTID, proj);

} // fillEB()
