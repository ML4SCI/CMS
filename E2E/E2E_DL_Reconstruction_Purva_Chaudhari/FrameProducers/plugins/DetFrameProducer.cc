//DetFrameProducer
#include "RecoE2E/FrameProducers/interface/DetFrameProducer.h"


DetFrameProducer::DetFrameProducer(const edm::ParameterSet& iConfig)
{
  EBRecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEBRecHitCollection"));
  photonCollectionT_ = consumes<PhotonCollection>(iConfig.getParameter<edm::InputTag>("photonCollection"));
  HBHERecHitCollectionT_  = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedHBHERecHitCollection"));
  EERecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEERecHitCollection"));
  trackCollectionT_       = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackCollection"));
  vertexCollectionT_       = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  TRKRecHitCollectionT_   = consumes<TrackingRecHitCollection>(iConfig.getParameter<edm::InputTag>("trackRecHitCollection"));
  genParticleCollectionT_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticleCollection"));
  pfCollectionT_          = consumes<PFCollection>(iConfig.getParameter<edm::InputTag>("pfCollection"));
  pvCollectionT_ = consumes<PVCollection>(iConfig.getParameter<edm::InputTag>("pvCollection"));
  jetTagCollectionT_      = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("jetTagCollection"));
  ipTagInfoCollectionT_   = consumes<std::vector<reco::CandIPTagInfo> > (iConfig.getParameter<edm::InputTag>("ipTagInfoCollection"));
  siPixelRecHitCollectionT_ = consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("siPixelRecHitCollection"));
  siStripRecHitCollectionT_ = consumes<SiStripMatchedRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("siStripMatchedRecHitCollection"));
  siStripMatchedRecHitCollectionT_ = consumes<SiStripMatchedRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("siStripMatchedRecHitCollection")); 
  ///////////adjustable granularity stuff

  granularityMultiPhi[0]  = iConfig.getParameter<int>("granularityMultiPhi");
  granularityMultiEta[0]  = iConfig.getParameter<int>("granularityMultiEta");

  granularityMultiPhi[1] = 3;
  granularityMultiEta[1] = 3;
  
  for (unsigned int proj=0; proj<Nadjproj; proj++)
  {
  
    int totalMultiEta = granularityMultiEta[proj] * granularityMultiECAL;

    for (int i=0; i<eta_nbins_HBHE; i++)
    {
      double step=(eta_bins_HBHE[i+1]-eta_bins_HBHE[i])/totalMultiEta;
      for (int j=0; j<totalMultiEta; j++)
      {
        adjEtaBins[proj].push_back(eta_bins_HBHE[i]+step*j);
      }
    }
    adjEtaBins[proj].push_back(eta_bins_HBHE[eta_nbins_HBHE]);

    totalEtaBins[proj] = totalMultiEta*(eta_nbins_HBHE);
    totalPhiBins[proj] = granularityMultiPhi[proj] * granularityMultiECAL*HBHE_IPHI_NUM;

  }
	
  // Detector image switches
  doECALstitched = iConfig.getParameter<bool>("doECALstitched");
  doTracksAtECALstitchedPt = iConfig.getParameter<bool>("doTracksAtECALstitchedPt");
  doTracksAtECALadjPt = iConfig.getParameter<bool>("doTracksAtECALadjPt");
  doHBHEenergy = iConfig.getParameter<bool>("doHBHEenergy");
  doBPIX1 = iConfig.getParameter<bool>("doBPIX1");
  doBPIX2 = iConfig.getParameter<bool>("doBPIX2");
  doBPIX3 = iConfig.getParameter<bool>("doBPIX3");
  doBPIX4 = iConfig.getParameter<bool>("doBPIX4");	
  doTOB = iConfig.getParameter<bool>("doTOB");
  doTIB = iConfig.getParameter<bool>("doTIB");
  doTEC = iConfig.getParameter<bool>("doTEC");
  doTID = iConfig.getParameter<bool>("doTID");
  setChannelOrder = iConfig.getParameter<std::string>("setChannelOrder");
  z0PVCut_   = iConfig.getParameter<double>("z0PVCut");

  //produces<float>("photonClasses").setBranchAlias("PhotonClass");
  produces<e2e::Frame1D>("EBenergy");
  produces<e2e::Frame1D>("HBHEenergyEB");
  produces<e2e::Frame1D>("TracksAtECALadj");
  produces<e2e::Frame1D>("TracksAtECALadjPtMax");
  produces<e2e::Frame1D>("ChannelMapping");
  produces<e2e::Frame3D>("DetFrames");
}

DetFrameProducer::~DetFrameProducer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DetFrameProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   //std::cout<<"doBPIX: "<<doBPIX<<std::endl;
   
   //std::cout<<"New Event started"<<std::endl;
   using namespace edm;
   nTotal++;
   // ----- Apply event selection cuts ----- //
   
   //auto photon_classes = std::make_unique<float>(10.0);
   fillEB( iEvent, iSetup );
   std::unique_ptr<e2e::Frame1D> EBenergy_edm (new e2e::Frame1D(vEB_energy_));
	
   //std::cout<<" >> Adding EB done "<<std::endl;
   //std::cout<<" >> Size of EB Energy vector is: "<<std::move(EBenergy_edm).get()->size()<<std::endl;
   edm::LogInfo("DetFrameProducer") << " >> Adding EB done.";
   edm::LogInfo("DetFrameProducer") << " >> Size of EB Energy vector is: " << std::move(EBenergy_edm).get()->size();
   iEvent.put(std::move(EBenergy_edm),"EBenergy");
	
   e2e::Frame3D vDetFrames;
   e2e::Frame3D vDetFrames_reordered;
   e2e::Frame2D vHBHE_energy_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vECAL_energy_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vECAL_tracksPt_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vECALadj_tracksPt_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vECAL_tracksd0_PV_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vECAL_tracksdz_PV_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vBPIX1_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vBPIX2_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vBPIX3_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vBPIX4_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vBPIX5_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vTOB_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vTIB_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vTEC_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame2D vTID_ECAL_reshaped (nDetFrameH, e2e::Frame1D (nDetFrameW,0.));
   e2e::Frame1D vChannelMap = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   
   if (doHBHEenergy){
	vChannelMap[0] = 1;
   	fillHBHE (iEvent, iSetup );
	// reshape detector image arrays to 280x360
	for (unsigned int idx=0; idx<vHBHE_energy_.size(); idx++){
		vHBHE_energy_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vHBHE_energy_[idx];
	}
   	std::unique_ptr<e2e::Frame1D> HBHEenergy_edm (new e2e::Frame1D(vHBHE_energy_));
   	std::unique_ptr<e2e::Frame1D> HBHEenergyEB_edm (new e2e::Frame1D(vHBHE_energy_EB_));
   	//std::cout<<" >> Size of HBHE Energy vector is: "<<std::move(HBHEenergy_edm).get()->size()<<std::endl;
   	//std::cout<<" >> Size of EB HBHE Energy vector is: "<<std::move(HBHEenergyEB_edm).get()->size()<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Size of HBHE Energy vector is: " << std::move(HBHEenergy_edm).get()->size();
	edm::LogInfo("DetFrameProducer") << " >> Size of EB HBHE Energy vector is: " << std::move(HBHEenergyEB_edm).get()->size();
	//iEvent.put(std::move(HBHEenergy_edm),"HBHEenergy");
   	//iEvent.put(std::move(HBHEenergyEB_edm),"HBHEenergyEB");
	vDetFrames.push_back(vHBHE_energy_reshaped);
   }
   
   if (doECALstitched){
	vChannelMap[1] = 1;
	fillECALstitched (iEvent, iSetup);
	// reshape detector image arrays to 280x360
	for (unsigned int idx=0; idx<vECAL_energy_.size(); idx++){
		vECAL_energy_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vECAL_energy_[idx];
	}
   	std::unique_ptr<e2e::Frame1D> ECALstitched_energy_edm (new e2e::Frame1D(vECAL_energy_));
   	//std::cout<<" >> Size of Stitched ECAL Energy vector is: "<<std::move(ECALstitched_energy_edm).get()->size()<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Size of Stitched ECAL Energy vector is: " <<std::move(ECALstitched_energy_edm).get()->size();
	//iEvent.put(std::move(ECALstitched_energy_edm), "ECALstitchedenergy");
   	vDetFrames.push_back(vECAL_energy_reshaped);
   }

   if (doTracksAtECALstitchedPt){
   	vChannelMap[2] = 1;
	vChannelMap[3] = 1;
        vChannelMap[4] = 1;
        fillTracksAtECALstitched (iEvent, iSetup );
	// reshape detector image arrays to 280x360
	for (unsigned int idx=0; idx<vECAL_tracksPt_.size(); idx++){
		vECAL_tracksPt_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vECAL_tracksPt_[idx];
	        vECAL_tracksd0_PV_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vECAL_tracksd0_PV_[idx];
                vECAL_tracksdz_PV_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vECAL_tracksdz_PV_[idx];
        }
   	std::unique_ptr<e2e::Frame1D> TracksECALstitchedPt_edm (new e2e::Frame1D(vECAL_tracksPt_));
   	//std::cout<<" >> Size of Pt Tracks vector at Stitched ECAL is: "<<std::move(TracksECALstitchedPt_edm).get()->size()<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Size of Pt Tracks vector at Stitched ECAL is: " << std::move(TracksECALstitchedPt_edm).get()->size();	
	//iEvent.put(std::move(TracksECALstitchedPt_edm), "TracksAtECALstitchedPt");
   	vDetFrames.push_back(vECAL_tracksPt_reshaped);
        vDetFrames.push_back(vECAL_tracksd0_PV_reshaped);
        vDetFrames.push_back(vECAL_tracksdz_PV_reshaped);
   }

   if (doTracksAtECALadjPt){
   	vChannelMap[5] = 1;
	for (unsigned int i=0;i<Nadjproj;i++)
   	{
     		fillTracksAtECALadjustable( iEvent, iSetup, i );
     		//fillTRKlayersAtECALadjustable( iEvent, iSetup, i );
   	}
	// reshape detector image arrays to 280x360
	for (unsigned int idx=0; idx<vECALadj_tracksPt_[0].size(); idx++){
		vECALadj_tracksPt_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vECALadj_tracksPt_[0][idx];
	}
   	//std::cout<<" >> Number of TracksAtECALadjPt per event: "<<sizeof(vECALadj_tracksPt_)/sizeof(vECALadj_tracksPt_[0])<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Number of TracksAtECALadjPt per event: " << sizeof(vECALadj_tracksPt_)/sizeof(vECALadj_tracksPt_[0]);
	//std::cout<<" >> Number of TracksAtECALadj per event: "<<sizeof(vECALadj_tracks_)/sizeof(vECALadj_tracks_[0])<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Number of TracksAtECALadj per event: " << sizeof(vECALadj_tracks_)/sizeof(vECALadj_tracks_[0]);
	//std::cout<<" >> Number of TracksAtECALadjPtMax per event: "<<sizeof(vECALadj_tracksPt_max_)/sizeof(vECALadj_tracksPt_max_[0])<<std::endl;
	edm::LogInfo("DetFrameProducer") << " >> Number of TracksAtECALadjPtMax per event: " << sizeof(vECALadj_tracksPt_max_)/sizeof(vECALadj_tracksPt_max_[0]);
   	//std::cout<<" >> Sizes of TracksadjPt, Tracksadj and TracksadjPtMax are: "<<vECALadj_tracksPt_[0].size()<<", "<<vECALadj_tracks_[0].size()<<", "<<vECALadj_tracksPt_max_[0].size()<<std::endl;
	edm::LogInfo("DetFrameProducer") << " >> Size of TracksECALadjPt, TracksECALadj and TracksECALadjPtMax are: "<< vECALadj_tracksPt_[0].size() << ", " << vECALadj_tracks_[0].size() <<", " << vECALadj_tracksPt_max_[0].size();
   	std::unique_ptr<e2e::Frame1D> TracksECALadjPt_edm (new e2e::Frame1D(vECALadj_tracksPt_[0]));
   	//std::cout<<" >> Size of Pt Tracks vector at ECAL adjustable is : "<<std::move(TracksECALadjPt_edm).get()->size()<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Size of Pt Tracks vector at ECAL adjustable is: " << std::move(TracksECALadjPt_edm).get()->size();
	//iEvent.put(std::move(TracksECALadjPt_edm),"TracksAtECALadjPt");
   	std::unique_ptr<e2e::Frame1D> TracksECALadj_edm (new e2e::Frame1D(vECALadj_tracks_[0]));
   	//std::cout<<" >> Size of Track vector at ECAL adjustable is : "<<std::move(TracksECALadj_edm).get()->size()<<std::endl;
	edm::LogInfo("DetFrameProducer") << " >> Size of Track vector at ECAL adjustable is: " << std::move(TracksECALadj_edm).get()->size();
   	//iEvent.put(std::move(TracksECALadj_edm),"TracksAtECALadj");
   	std::unique_ptr<e2e::Frame1D> TracksECALadjPt_max_edm (new e2e::Frame1D(vECALadj_tracksPt_max_[0]));
   	//std::cout<<" >> Size of max Pt Track vector at ECAL adjustable is : "<<std::move(TracksECALadjPt_max_edm).get()->size()<<std::endl;
   	edm::LogInfo("DetFrameProducer") << " >> Size of max Pt Track vector at ECAL adjustable is: " << std::move(TracksECALadjPt_max_edm).get()->size();
	//iEvent.put(std::move(TracksECALadjPt_max_edm),"TracksAtECALadjPtMax");
   	vDetFrames.push_back(vECALadj_tracksPt_reshaped);
   }
   //std::cout<<"doBPIX: "<<doBPIX<<std::endl;
   if (doBPIX1){
	     vChannelMap[6] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedBPIX( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]); idx++){
        	vBPIX1_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vBPIX_ECAL_[0][0][idx];
        }
	vDetFrames.push_back(vBPIX1_ECAL_reshaped);
	//std::cout<<" >> Number of BPIX channels: "<<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0])<<" and Size of each BPIX channel: "<<vBPIX_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of BPIX channels: " << sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]) << " and Size of each BPIX channel: " << vBPIX_ECAL_[0][0].size();
   }
   if (doBPIX2){
	     vChannelMap[7] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedBPIX( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]); idx++){
		   vBPIX2_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vBPIX_ECAL_[1][0][idx];
        }
	vDetFrames.push_back(vBPIX2_ECAL_reshaped);
	//std::cout<<" >> Number of BPIX channels: "<<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0])<<" and Size of each BPIX channel: "<<vBPIX_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of BPIX channels: " << sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]) << " and Size of each BPIX channel: " << vBPIX_ECAL_[0][0].size();
   }

   if (doBPIX3){
	     vChannelMap[8] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedBPIX( iEvent, iSetup, i );
         
  	}
       // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]); idx++){
		      vBPIX3_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vBPIX_ECAL_[2][0][idx];
        }

	vDetFrames.push_back(vBPIX3_ECAL_reshaped);
	//std::cout<<" >> Number of BPIX channels: "<<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0])<<" and Size of each BPIX channel: "<<vBPIX_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of BPIX channels: " << sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]) << " and Size of each BPIX channel: " << vBPIX_ECAL_[0][0].size();
   }

   if (doBPIX4){
        vChannelMap[9] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedBPIX( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]); idx++){
        	
		      vBPIX4_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vBPIX_ECAL_[3][0][idx];
        }

	vDetFrames.push_back(vBPIX4_ECAL_reshaped);
	//std::cout<<" >> Number of BPIX channels: "<<sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0])<<" and Size of each BPIX channel: "<<vBPIX_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of BPIX channels: " << sizeof(vBPIX_ECAL_)/sizeof(vBPIX_ECAL_[0]) << " and Size of each BPIX channel: " << vBPIX_ECAL_[0][0].size();
   }

 
   if (doTOB){
        vChannelMap[11] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedTOB( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vTOB_ECAL_)/sizeof(vTOB_ECAL_[0]); idx++){
        	
		vTOB_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vTOB_ECAL_[0][0][idx];
        }
	vDetFrames.push_back(vTOB_ECAL_reshaped);
	//std::cout<<" >> Number of TOB channels: "<<sizeof(vTOB_ECAL_)/sizeof(vTOB_ECAL_[0])<<" and Size of each TOB channel: "<<vTOB_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of TOB channels: " << sizeof(vTOB_ECAL_)/sizeof(vTOB_ECAL_[0]) << " and Size of each TOB channel: " << vTOB_ECAL_[0][0].size();
   }
     if (doTIB){
        vChannelMap[12] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedTIB( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vTIB_ECAL_)/sizeof(vTIB_ECAL_[0]); idx++){
        	
		vTIB_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vTIB_ECAL_[4][0][idx];
        }
	vDetFrames.push_back(vTIB_ECAL_reshaped);
	//std::cout<<" >> Number of TIB channels: "<<sizeof(vTIB_ECAL_)/sizeof(vTIB_ECAL_[0])<<" and Size of each TIB channel: "<<vTIBB_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of TIB channels: " << sizeof(vTIB_ECAL_)/sizeof(vTIB_ECAL_[0]) << " and Size of each TIB channel: " << vTIB_ECAL_[0][0].size();
   }
     if (doTEC){
        vChannelMap[13] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedTEC( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vTEC_ECAL_)/sizeof(vTEC_ECAL_[0]); idx++){
        	
		vTEC_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vTEC_ECAL_[4][0][idx];
        }
	vDetFrames.push_back(vTEC_ECAL_reshaped);
	//std::cout<<" >> Number of TEC channels: "<<sizeof(vTEC_ECAL_)/sizeof(vBPIX_ECAL_[0])<<" and Size of each TEC channel: "<<vTEC_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of BPIX channels: " << sizeof(vTEC_ECAL_)/sizeof(vTEC_ECAL_[0]) << " and Size of each TEC channel: " << vTEC_ECAL_[0][0].size();
   }
     if (doTID){
        vChannelMap[14] = 1;
	for (unsigned int i=0;i<Nhitproj;i++)
  	{
    	  fillTRKlayersAtECALstitchedTID( iEvent, iSetup, i );
  	}
        // reshape detector image arrays to 280x360
        for (unsigned int idx=0; idx<sizeof(vTID_ECAL_)/sizeof(vTID_ECAL_[0]); idx++){
        	
		vTID_ECAL_reshaped[int(idx/nDetFrameW)][idx%nDetFrameW]=vTID_ECAL_[4][0][idx];
        }
	vDetFrames.push_back(vTID_ECAL_reshaped);
	//std::cout<<" >> Number of TID channels: "<<sizeof(vTID_ECAL_)/sizeof(vTID_ECAL_[0])<<" and Size of each TID channel: "<<vTID_ECAL_[0][0].size()<<std::endl; 
   	edm::LogInfo("DetFrameProducer") << " >> Number of TID channels: " << sizeof(vTID_ECAL_)/sizeof(vTID_ECAL_[0]) << " and Size of each TID channel: " << vTID_ECAL_[0][0].size();
   }

   // Convert string of channel order to array of integers
   int str_start = 0;
   std::string del = ",";
   int str_end = setChannelOrder.find(del);
   int channel_order[vDetFrames.size()]; //int channel_order[] = {2,0,1,3,4,5,7,6};
   int ch_idx=0;
   while (str_end !=-1){
	channel_order[ch_idx] = ((int)(char) (setChannelOrder.substr(str_start, str_end - str_start)[0])) - (int)('0');
	str_start = str_end + del.size();
	str_end = setChannelOrder.find(del,str_start);
	ch_idx++;
   } 
   channel_order[ch_idx] = ((int)(char) (setChannelOrder.substr(str_start, str_end-str_start)[0])) - (int)('0');
   /*
   for (unsigned int i=0;i<vDetFrames.size();i++){
	std::cout<<channel_order[i]<<" ";
   }
   std::cout<<std::endl;
   */
   
   // Reorder Channels
   if (sizeof(setChannelOrder)/sizeof(setChannelOrder[0]) != vDetFrames.size()) {
	edm::LogInfo("DetFrameProducer") << " !! Channel ordering size invalid. Ignoring the order. !! ";
   	for (unsigned int ch_idx=0;ch_idx<vDetFrames.size();ch_idx++){
		vDetFrames_reordered.push_back(vDetFrames[ch_idx]);
	}
   } 
   else {
	for (unsigned int ch_idx=0;ch_idx<sizeof(channel_order)/sizeof(channel_order[0]);ch_idx++){
		vDetFrames_reordered.push_back(vDetFrames[channel_order[ch_idx]]);
   	}
   }

   //std::cout<<"vDet size: "<<vDetFrames_reordered.size()<<std::endl;
   edm::LogInfo("DetFrameProducer") << " >> vDet size: " << vDetFrames_reordered.size();
   std::unique_ptr<e2e::Frame3D> vDetFrames_edm (new e2e::Frame3D(vDetFrames_reordered));
   iEvent.put(std::move(vDetFrames_edm), "DetFrames");

   std::unique_ptr<e2e::Frame1D> vChannelMap_edm (new e2e::Frame1D(vChannelMap));
   iEvent.put(std::move(vChannelMap_edm), "ChannelMapping");
  
   //std::cout<<" >> Added EB, HBHE, HBHE_EB, ECALstitched, TracksAtECALstitchedPt and TracksAtECALadjPt to edm root file"<<std::endl;
   edm::LogInfo("DetFrameProducer") << " >> Adding selected channels to edm root file.";

   nPassed++;
   return;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
DetFrameProducer::beginStream(edm::StreamID)
{
 nTotal = 0;
 nPassed = 0;
 //std::cout<<"'DetFrameProducer' Stream began"<<std::endl;
 edm::LogInfo("DetFrameProducer") << " ' DetFrameProducer' Stream began";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
DetFrameProducer::endStream() {
 //std::cout << "'DetFrameProducer' selected: " << nPassed << "/" << nTotal << std::endl;
 edm::LogInfo("DetFrameProducer") << " 'DetFrameProducer' selected: " << nPassed << "/" << nTotal;
}

// ------------ method called when starting to processes a run  ------------
/*
void
ProducerTest::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
ProducerTest::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
ProducerTest::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
ProducerTest::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DetFrameProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


//define this as a plug-in
DEFINE_FWK_MODULE(DetFrameProducer);
