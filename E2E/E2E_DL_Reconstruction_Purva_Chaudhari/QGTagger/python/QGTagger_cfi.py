import FWCore.ParameterSet.Config as cms

QGTagger = cms.EDProducer('QGTagger'
    , QGFrames = cms.InputTag('JetFrames','JetFrames')
    , reducedHBHERecHitCollection = cms.InputTag('reducedHcalRecHits:hbhereco')
    , ak4PFJetCollection = cms.InputTag('ak4PFJetsCHS')
    , ak4GenJetCollection = cms.InputTag('ak4GenJets')
    , ak4RecoJetsForBTagging = cms.InputTag("ak4PFJetsCHS")
    , mode = cms.string("JetLevel")
    
    , siPixelRecHitCollection = cms.InputTag('siPixelRecHits')
    , siStripRecHitCollection =  cms.VInputTag(
    cms.InputTag('siStripMatchedRecHits:rphiRecHit'),
    cms.InputTag('siStripMatchedRecHits:stereoRecHit'),
    cms.InputTag('siStripMatchedRecHits:rphiRecHitUnmatched'),
    cms.InputTag('siStripMatchedRecHits:stereoRecHitUnmatched')
    )

    # Jet level cfg
    , nJets = cms.int32(2)
    , minJetPt = cms.double(400.)
    , maxJetEta = cms.double(1.37)
    , z0PVCut  = cms.double(0.1)
    , isTTbar = cms.bool(True)
    , minTopPt = cms.double(200.)
    , maxTopEta = cms.double(2.4)
    , doECALstitched = cms.bool(True)
    , doTracksAtECALstitchedPt = cms.bool(True)
    , doTracksAtECALadjPt = cms.bool(True)
    , doHBHEenergy = cms.bool(True)
    , QGModelName = cms.string('tfModels/ResNet_4_channel_tf13.pb')
    )
