#include "RecoE2E/FrameProducers/interface/JetFrameProducer.h"

JetFrameProducer::JetFrameProducer(const edm::ParameterSet& iConfig)
{
  photonCollectionT_ = consumes<PhotonCollection>(iConfig.getParameter<edm::InputTag>("photonCollection"));
  EBRecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEBRecHitCollection"));
  HBHERecHitCollectionT_  = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedHBHERecHitCollection"));
  vDetFramesT_ = consumes<e2e::Frame3D>(iConfig.getParameter<edm::InputTag>("DetFrames"));
  vertexCollectionT_       = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  
  TRKRecHitCollectionT_   = consumes<TrackingRecHitCollection>(iConfig.getParameter<edm::InputTag>("trackRecHitCollection"));
  genParticleCollectionT_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticleCollection"));
  
  pfCollectionT_          = consumes<PFCollection>(iConfig.getParameter<edm::InputTag>("pfCollection"));
  
  jetTagCollectionT_      = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("jetTagCollection"));
  ipTagInfoCollectionT_   = consumes<std::vector<reco::CandIPTagInfo> > (iConfig.getParameter<edm::InputTag>("ipTagInfoCollection"));
  
  // Jet Collection switches
  jetCollection_sel = iConfig.getParameter<std::string>("jetCollection");//
  if (jetCollection_sel == "ak4"){
    jetCollectionT_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("ak4PFJetCollection"));
    genJetCollectionT_      = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("ak4GenJetCollection"));
    recoJetsT_              = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("ak4RecoJetsForBTagging"));
  }
  else if (jetCollection_sel == "ak8"){
    jetCollectionT_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("ak8PFJetCollection"));
    genJetCollectionT_      = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("ak8GenJetCollection"));
    recoJetsT_              = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("ak8RecoJetsForBTagging"));
  }
  
  mode_      = iConfig.getParameter<std::string>("mode");
  minJetPt_  = iConfig.getParameter<double>("minJetPt");
  maxJetEta_ = iConfig.getParameter<double>("maxJetEta");
  z0PVCut_   = iConfig.getParameter<double>("z0PVCut");
  
  // Detector image switches
  doECALstitched = iConfig.getParameter<bool>("doECALstitched");
  doTracksAtECALstitchedPt = iConfig.getParameter<bool>("doTracksAtECALstitchedPt");
  doTracksAtECALadjPt = iConfig.getParameter<bool>("doTracksAtECALadjPt");
  doHBHEenergy = iConfig.getParameter<bool>("doHBHEenergy");
	
  // Output collections to be produced
  produces<e2e::Frame2D> ("JetSeeds");
  produces<e2e::Frame4D> ("JetFrames");
}

JetFrameProducer::~JetFrameProducer()
{
}

void
JetFrameProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::LogInfo("JetFrameProducer") << " >> Running JetFrameProducer...";
  
  nTotal++;
  
  // Selecting Jet Seeds (ak8 / ak4) and storing them in edm root file.
  edm::LogInfo("JetFrameProducer") << " >> doJets set";
  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(jetCollectionT_, jets);
  if ( debug ) std::cout << " >> PFJetCol.size: " << jets->size() << std::endl;
  e2e::Frame2D    vJetSeeds ( jets->size(), std::vector<float> (nSeedCoords, float(defaultVal)) );
  //std::cout<<"Raw seeds are: ";
  edm::LogInfo("JetFrameProducer") << " >> Raw seeds are: ";
  for (int iJ = 0; iJ<int(jets->size()); iJ++){
	reco::PFJetRef iJet( jets, iJ );
  	//std::cout<<"("<<iJet->eta()<<","<<iJet->phi()<<")";
  	edm::LogInfo("JetFrameProducer") << "(" << iJet->eta() << "," <<iJet->phi() << ")";
  }
  //std::cout<<std::endl;
  getJetseed( iEvent, iSetup, vJetSeeds );
  //std::cout<<" >> The seeds are: ";
  edm::LogInfo("JetFrameProducer") << " >> The seeds are: ";
  for (int idx=0; idx<int(vJetSeeds.size());idx++){
  	//std::cout<<"("<<vJetSeeds[idx][0]<<","<<vJetSeeds[idx][1]<<")";
  	edm::LogInfo("JetFrameProducer") << "(" << vJetSeeds[idx][0] << "," << vJetSeeds[idx][1] << ")";
  }
  //std::cout<<std::endl;
  //std::cout<<" >> Number of Jets: "<<vJetSeeds.size()<<std::endl;
  edm::LogInfo("JetFrameProducer") << " >> Number of Jets: " << vJetSeeds.size();
  //std::cout<<" >> The jet seeds are (ieta,iphi): ";
  edm::LogInfo("JetFrameProducer") << " >> The jet seeds are (ieta,iphi): ";
    
  for (int idx=0;idx<int(vJetSeeds.size());idx++){
	//std::cout<<"("<<vJetSeeds[idx][0]<<","<<vJetSeeds[idx][1]<<") ";
     	edm::LogInfo("JetFrameProducer") << "(" << vJetSeeds[idx][0] << "," << vJetSeeds[idx][1] << ") ";
	if(vJetSeeds[idx][0]>=0){vJetSeeds[idx][0]=int(vJetSeeds[idx][0]*5+2);}  //5 EB xtals per HB tower
	if(vJetSeeds[idx][1]>=0){vJetSeeds[idx][1]=int(vJetSeeds[idx][1]*5+2);}  //5 EB xtals per HB tower
   }
   //std::cout<<std::endl;
   
   edm::Handle<e2e::Frame3D> vDetFrames_handle;
   iEvent.getByToken(vDetFramesT_, vDetFrames_handle);
   e2e::Frame3D vDetFrames = *vDetFrames_handle;
  
   //Performing Striding on HBHE Frames.
   if(doHBHEenergy) vDetFrames[0] = frameStriding(vDetFrames[0], int(nDetImgH/nStrideH), int(nDetImgW/nStrideW), nStrideH, nStrideW);
	
   // Put collections into output EDM file
   std::unique_ptr<e2e::Frame2D> cJetSeeds (new e2e::Frame2D(vJetSeeds));
   iEvent.put( std::move(cJetSeeds),  "JetSeeds"  );
   
   unsigned int nFrameD = vDetFrames.size();	
   std::vector<e2e::Frame3D> vJetFrames (vJetSeeds.size(),
				      e2e::Frame3D(nFrameD,
				      e2e::Frame2D(nFrameH,
				      e2e::Frame1D(nFrameW, 0.))) );	
	
   for (int idx=0;idx<int(vJetSeeds.size());idx++){
   	//std::cout<<"Generating stitched and adjustable ECAL frames and their track frames from the jet seed "<<idx+1<<"/"<<vJetSeeds.size()<<" with seed value: ("<<vJetSeeds[idx][0]<<","<<vJetSeeds[idx][1]<<")"<<std::endl;
   	edm::LogInfo("JetFrameProducer") << "Generating stitched and adjustable ECAL frames and their track frames from the jet seed " << idx+1 << vJetSeeds.size() << " with seed value: (" << vJetSeeds[idx][0] << "," << vJetSeeds[idx][1] << ")";
	if(vJetSeeds[idx][0]>=0) {
   		// Producing cropped frames from the seeds.
		e2e::seed vJetSeed_ = {-1,-1};
		vJetSeed_[0] = vJetSeeds[idx][0];
		vJetSeed_[1] = vJetSeeds[idx][1];
		for (int layer_idx=0; layer_idx<int(nFrameD); layer_idx++){
			e2e::getFrame(vJetFrames[idx][layer_idx], vJetSeed_, &vDetFrames[layer_idx], nDetImgH, nDetImgW);
		}	
	}
   }
   
   //e2e::Frame4D tmp = vECALstitchedFrames;
   //e2e::Frame2D tmp_out = e2e::predict_tf(vECALstitchedFrames, "ResNet.pb", "inputs","outputs");
   // Put collections into output EDM file
   std::unique_ptr<e2e::Frame4D> cJetFrames (new e2e::Frame4D(vJetFrames));
   iEvent.put( std::move(cJetFrames),  "JetFrames"  ); 
   nPassed++;
   return;
}

void JetFrameProducer::getJetseed ( const edm::Event& iEvent, const edm::EventSetup& iSetup, e2e::Frame2D& vJetSeeds )
{
	vector<int>   vFailedJetIdx_;
	const int search_window = 7;
	const int image_padding = 12;
	
	edm::ESHandle<CaloGeometry> caloGeomH_;
   	iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
   	const CaloGeometry* caloGeom = caloGeomH_.product();
	
   	edm::Handle<HBHERecHitCollection> HBHERecHitsH_;
   	iEvent.getByToken( HBHERecHitCollectionT_, HBHERecHitsH_ );
	
   	edm::Handle<reco::PFJetCollection> jets;
   	iEvent.getByToken(jetCollectionT_, jets);
   	if ( debug ) std::cout << " >> PFJetCol.size: " << jets->size() << std::endl;
   
	float seedE;
   	int iphi_, ieta_, ietaAbs_;
   	int nJet = 0;
	//std::cout<<" >> Reading and selecting Jets from "<<jets->size()<<" jet seeds: "<<std::endl;
	edm::LogInfo("JetFrameProducer") << " >> Reading and selecting Jets from " << jets->size() << " jet seeds: ";   	

	for (unsigned iJ=0;iJ<jets->size();iJ++){
    	iphi_ = -1;
    	ieta_ = -1;
	reco::PFJetRef iJet( jets, iJ );
		
	if ( debug ) std::cout << " >> jet[" << iJ << "]Pt:" << iJet->pt()  << " Eta:" << iJet->eta()  << " Phi:" << iJet->phi() 
			   << " jetE:" << iJet->energy() << " jetM:" << iJet->mass() << std::endl;
	HcalDetId hId( spr::findDetIdHCAL( caloGeom, iJet->eta(), iJet->phi(), false ) );
    	if ( hId.subdet() != HcalBarrel && hId.subdet() != HcalEndcap ){
		//std::cout<<"     * Failed Jet index because hId.subdet not equal to HCAL Barrel and HCAL Endcap. Adding -1 to Jet Seed vector"<<std::endl;
      		edm::LogInfo("JetFrameProducer") << "     * Failed Jet index because hId.subdet not equal to HCAL Barrel and HCAL Endcap. Adding -1 to Jet Seed vector.";
		vFailedJetIdx_.push_back(iJ);
      		continue;
    	}
	   
	HBHERecHitCollection::const_iterator iRHit( HBHERecHitsH_->find(hId) );
    	seedE = ( iRHit == HBHERecHitsH_->end() ) ? 0. : iRHit->energy() ;
    	HcalDetId seedId = hId;
    	if ( debug ) std::cout << " >> hId.ieta:" << hId.ieta() << " hId.iphi:" << hId.iphi() << " E:" << seedE << std::endl;
	
	// Look for the most energetic HBHE tower deposit within a search window
	for ( int ieta = 0; ieta < search_window; ieta++ ) {

      		ieta_ = hId.ieta() - (search_window/2)+ieta;

      		if ( std::abs(ieta_) > HBHE_IETA_MAX_HE-1 ) continue;
      		if ( std::abs(ieta_) < HBHE_IETA_MIN_HB ) continue;

      		HcalSubdetector subdet_ = std::abs(ieta_) > HBHE_IETA_MAX_HB ? HcalEndcap : HcalBarrel;

      		for ( int iphi = 0; iphi < search_window; iphi++ ) {

        		iphi_ = hId.iphi() - (search_window/2)+iphi;

        		// iphi should wrap around
        		if ( iphi_ > HBHE_IPHI_MAX ) {
          		iphi_ = iphi_-HBHE_IPHI_MAX;
        		} else if ( iphi_ < HBHE_IPHI_MIN ) {
          		iphi_ = HBHE_IPHI_MAX-abs(iphi_); 
        		}
        		// Skip non-existent and lower energy towers 
        		HcalDetId hId_( subdet_, ieta_, iphi_, 1 );
        		HBHERecHitCollection::const_iterator iRHit( HBHERecHitsH_->find(hId_) );
        		if ( iRHit == HBHERecHitsH_->end() ) continue;
        		if ( iRHit->energy() <= seedE ) continue;
        		if ( debug ) std::cout << " !! hId.ieta:" << hId_.ieta() << " hId.iphi:" << hId_.iphi() << " E:" << iRHit->energy() << std::endl;

        		seedE = iRHit->energy();
        		seedId = hId_;

      		} // iphi 
    	} // ieta
	
    	// NOTE: HBHE iphi = 1 does not correspond to EB iphi = 1!
    	// => Need to shift by 2 HBHE towers: HBHE::iphi: [1,...,71,72]->[3,4,...,71,72,1,2]
    	iphi_  = seedId.iphi() + 2; // shift
    	iphi_  = iphi_ > HBHE_IPHI_MAX ? iphi_-HBHE_IPHI_MAX : iphi_; // wrap-around
    	iphi_  = iphi_ - 1; // make histogram-friendly
    	ietaAbs_  = seedId.ietaAbs() == HBHE_IETA_MAX_HE ? HBHE_IETA_MAX_HE-1 : seedId.ietaAbs(); // last HBHE ieta embedded
    	ieta_  = seedId.zside() > 0 ? ietaAbs_-1 : -ietaAbs_;
    	ieta_  = ieta_+HBHE_IETA_MAX_HE-1;

    	// If the seed is too close to the edge of HE, discard event
    	// Required to keep the seed at the image center
    	if ( HBHE_IETA_MAX_HE-1 - ietaAbs_ < image_padding ) { 
      	if ( debug ) std::cout << " Fail HE edge cut " << std::endl;
      	vFailedJetIdx_.push_back(iJ);
	//std::cout<<"     * Failed Jet seed at index: "<<iJ<<" seed is too close to the edge of HE. Adding -1 to Jet Seed vectors."<<std::endl;
	edm::LogInfo("JetFrameProducer") << "     * Failed Jet seed at index: " << iJ << " seed is too close to the edge of HE. Adding -1 to Jet Seed vectors.";
	ieta_=-1;
	iphi_=-1;
    	nJet++;
      	continue;
    	}
    	// Save position of most energetic HBHE tower
    	// in EB-aligned coordinates
    	if ( debug ) std::cout << " !! ieta_:" << ieta_ << " iphi_:" << iphi_ << " ietaAbs_:" << ietaAbs_ << " E:" << seedE << std::endl;
	
    	vJetSeeds[iJ][0] = ieta_;
	vJetSeeds[iJ][1] = iphi_;
    	nJet++;
   } // good jets	
  
  return;
} // JetFrameProducer::getEGseed()


// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
JetFrameProducer::beginStream(edm::StreamID)
{
 nTotal = 0;
 nPassed = 0;
 //std::cout<<"'JetFrameProducer' Stream initiated"<<std::endl;
 edm::LogInfo("JetFrameProducer") << " 'JetFrameProducer' Stream initiated."; 
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
JetFrameProducer::endStream() {
 //std::cout << "'JetFrameProducer' selected: " << nPassed << "/" << nTotal << std::endl;
 edm::LogInfo("JetFrameProducer") << " 'JetFrameProducer' selected: " << nPassed << "/" << nTotal;
}

// ------------ method called when starting to processes a run  ------------
/*
void
JetFrameProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
JetFrameProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
JetFrameProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
JetFrameProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetFrameProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
const reco::PFCandidate*
JetFrameProducer::getPFCand(edm::Handle<PFCollection> pfCands, float eta, float phi, float& minDr, bool debug ) {

  minDr = 10;
  const reco::PFCandidate* minDRCand = nullptr;
  
  for ( PFCollection::const_iterator iPFC = pfCands->begin();
        iPFC != pfCands->end(); ++iPFC ) {

    const reco::Track* thisTrk = iPFC->bestTrack();
    if ( !thisTrk ) continue;

    float thisdR = reco::deltaR( eta, phi, thisTrk->eta(), thisTrk->phi() );
    if (debug) std::cout << "\tthisdR: " << thisdR << " " << thisTrk->pt() << " " << iPFC->particleId() << std::endl;

    const reco::PFCandidate& thisPFCand = (*iPFC);
      
    if ( (thisdR < 0.01) && (thisdR <minDr) ) {
      minDr    = thisdR; 
      minDRCand = &thisPFCand;
    }
  }

  return minDRCand;  
}

const reco::Track*
JetFrameProducer::getTrackCand(edm::Handle<reco::TrackCollection> trackCands, float eta, float phi, float& minDr, bool debug ) {

  minDr = 10;
  const reco::Track* minDRCand = nullptr;
  reco::Track::TrackQuality tkQt_ = reco::Track::qualityByName("highPurity");

  for ( reco::TrackCollection::const_iterator iTk = trackCands->begin();
        iTk != trackCands->end(); ++iTk ) {
    if ( !(iTk->quality(tkQt_)) ) continue;  

    float thisdR = reco::deltaR( eta, phi, iTk->eta(),iTk->phi() );
    if (debug) std::cout << "\tthisdR: " << thisdR << " " << iTk->pt() << std::endl;

    const reco::Track& thisTrackCand = (*iTk);
      
    if ( (thisdR < 0.01) && (thisdR <minDr) ) {
      minDr    = thisdR; 
      minDRCand = &thisTrackCand;
    }
  }

  return minDRCand;  
}




int JetFrameProducer::getTruthLabel(const reco::PFJetRef& recJet, edm::Handle<reco::GenParticleCollection> genParticles, float dRMatch , bool debug ){
  if ( debug ) {
    std::cout << " Mathcing reco jetPt:" << recJet->pt() << " jetEta:" << recJet->eta() << " jetPhi:" << recJet->phi() << std::endl;
  }

  for (reco::GenParticleCollection::const_iterator iGen = genParticles->begin();
       iGen != genParticles->end();
       ++iGen) {

    // From: (page 7/ Table 1.5.2)
    //https://indico.desy.de/indico/event/7142/session/9/contribution/31/material/slides/6.pdf
    //code range explanation:
    // 11 - 19 beam particles
    // 21 - 29 particles of the hardest subprocess
    // 31 - 39 particles of subsequent subprocesses in multiparton interactions
    // 41 - 49 particles produced by initial-state-showers
    // 51 - 59 particles produced by final-state-showers
    // 61 - 69 particles produced by beam-remnant treatment
    // 71 - 79 partons in preparation of hadronization process
    // 81 - 89 primary hadrons produced by hadronization process
    // 91 - 99 particles produced in decay process, or by Bose-Einstein effects

    // Do not want to match to the final particles in the shower
    if ( iGen->status() > 99 ) continue;
    
    // Only want to match to partons/leptons/bosons
    if ( iGen->pdgId() > 25 ) continue;

    float dR = reco::deltaR( recJet->eta(),recJet->phi(), iGen->eta(),iGen->phi() );

    if ( debug ) std::cout << " \t >> dR " << dR << " id:" << iGen->pdgId() << " status:" << iGen->status() << " nDaught:" << iGen->numberOfDaughters() << " pt:"<< iGen->pt() << " eta:" <<iGen->eta() << " phi:" <<iGen->phi() << " nMoms:" <<iGen->numberOfMothers()<< std::endl;

    if ( dR > dRMatch ) continue; 
    if ( debug ) std::cout << " Matched pdgID " << iGen->pdgId() << std::endl;

    return iGen->pdgId();

  } // gen particles 





  return -99;
}


float JetFrameProducer::getBTaggingValue(const reco::PFJetRef& recJet, edm::Handle<edm::View<reco::Jet> >& recoJetCollection, edm::Handle<reco::JetTagCollection>& btagCollection, float dRMatch, bool debug ){

  // loop over jets
  for( edm::View<reco::Jet>::const_iterator jetToMatch = recoJetCollection->begin(); jetToMatch != recoJetCollection->end(); ++jetToMatch )
    {
      reco::Jet thisJet = *jetToMatch;
      float dR = reco::deltaR( recJet->eta(),recJet->phi(), thisJet.eta(),thisJet.phi() );
      if(dR > 0.1) continue;

      size_t idx = (jetToMatch - recoJetCollection->begin());
      edm::RefToBase<reco::Jet> jetRef = recoJetCollection->refAt(idx);

      if(debug) std::cout << "btag discriminator value = " << (*btagCollection)[jetRef] << std::endl;
      return (*btagCollection)[jetRef];
  
    }

  if(debug){
    std::cout << "ERROR  No btag match: " << std::endl;
    
    // loop over jets
    for( edm::View<reco::Jet>::const_iterator jetToMatch = recoJetCollection->begin(); jetToMatch != recoJetCollection->end(); ++jetToMatch )
      {
	const reco::Jet thisJet = *jetToMatch;
	std::cout << "\t Match attempt pt: " <<  thisJet.pt() << " vs " <<  recJet->pt()
		  << " eta: " << thisJet.eta() << " vs " << recJet->eta()
		  << "phi: "<< thisJet.phi() << " vs " << recJet->phi()
		  << std::endl;
	float dR = reco::deltaR( recJet->eta(),recJet->phi(), thisJet.eta(),thisJet.phi() );
	std::cout << "dR " << dR << std::endl;
      }
  }    

  return -99;
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetFrameProducer);
