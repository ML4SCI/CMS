#include "RecoE2E/TauTagger/interface/TauTagger.h"

const int search_window = 7;
const int image_padding = 12;
vector<int>   vFailedJetIdx_;
extern unsigned int jet_runId_;
extern unsigned int jet_lumiId_;
extern unsigned long long jet_eventId_;

void TauTagger::runEvtSel_jet ( const edm::Event& iEvent, const edm::EventSetup& iSetup, e2e::Frame2D& vJetSeeds ) {

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
   vFailedJetIdx_.clear();
   
   //std::cout<<" >> Reading and selecting Jets from "<<jets->size()<<" jet seeds: "<<std::endl;
   edm::LogInfo("TauTagger") << " >> Reading and selecting Jets from " << jets->size() << " jet seeds: ";
   for (unsigned iJ=0;iJ<jets->size();iJ++){
   	bool keepJet = true;
    	iphi_ = -1;
    	ieta_ = -1;
	reco::PFJetRef iJet( jets, iJ );
	   
	// Jet selection criteria
    	if ( std::abs(iJet->pt())  < minJetPt_ ) {keepJet = false; 
		//std::cout<<"     * Selection failed at Jet index: "<<iJ<<" because abs(pt) < minJetPt_ --> pt: "<<std::abs(iJet->pt())<<" minJetPt_: "<<minJetPt_<<". Adding "<<iphi_<<" to JetSeediphi and "<<ieta_<<" JetSeedieta vectors."<<std::endl;
		edm::LogInfo("TauTagger") << "     * Selection failed at Jet index: "<<iJ<<" because abs(pt) < minJetPt_ --> pt: "<<std::abs(iJet->pt())<<" minJetPt_: "<<minJetPt_<<". Adding "<<iphi_<<" to JetSeediphi and "<<ieta_<<" JetSeedieta vectors.";
    		nJet++;					 
	}
    	else if ( std::abs(iJet->eta()) > maxJetEta_ ) {keepJet = false; 
		//std::cout<<"     * Selection failed at Jet index: "<<iJ<<" because abs(eta) > maxJetEta_ --> eta: "<<std::abs(iJet->eta())<<" maxJetEta_: "<<maxJetEta_<<". Adding "<<iphi_<<" to JetSeediphi and "<<ieta_<<" JetSeedieta vectors."<<std::endl;
		edm::LogInfo("TauTagger") << "     * Selection failed at Jet index: "<<iJ<<" because abs(eta) > maxJetEta_ --> eta: "<<std::abs(iJet->eta())<<" maxJetEta_: "<<maxJetEta_<<". Adding "<<iphi_<<" to JetSeediphi and "<<ieta_<<" JetSeedieta vectors.";
    		nJet++;					  
	}
	//std::cout<<" # keepJet: "<<keepJet<<" --> pt: "<<std::abs(iJet->pt())<<", eta: "<<std::abs(iJet->eta())<<", minJetPt: "<<minJetPt_<<", maxJetEta: "<<maxJetEta_<<std::endl;
	if (keepJet){
	if ( debug ) std::cout << " >> jet[" << iJ << "]Pt:" << iJet->pt()  << " Eta:" << iJet->eta()  << " Phi:" << iJet->phi() 
			   << " jetE:" << iJet->energy() << " jetM:" << iJet->mass() << std::endl;
	HcalDetId hId( spr::findDetIdHCAL( caloGeom, iJet->eta(), iJet->phi(), false ) );
    	if ( hId.subdet() != HcalBarrel && hId.subdet() != HcalEndcap ){
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
	//std::cout<<"     * Failed Jet seed at index: "<<iJ<<" seed is too close to the edge of HE. Adding -1 to JetSeediphi and JetSeedieta vectors."<<std::endl;
	edm::LogInfo("TauTagger") << "     * Failed Jet seed at index: " << iJ << " seed is too close to the edge of HE. Adding -1 to JetSeediphi and JetSeedieta vectors.";	
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
      }
   } // good jets	
  
  return;

} // runEvtSel_jet()
