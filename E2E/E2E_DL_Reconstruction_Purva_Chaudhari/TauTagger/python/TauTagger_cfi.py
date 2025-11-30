import FWCore.ParameterSet.Config as cms

TauTagger = cms.EDProducer('TauTagger'
    , TauFrames = cms.InputTag('JetFrames','JetFrames')
    , reducedHBHERecHitCollection = cms.InputTag('reducedHcalRecHits:hbhereco')
    , ak8PFJetCollection = cms.InputTag('ak8PFJetsCHS')
    , ak8GenJetCollection = cms.InputTag('ak8GenJets')
    , ak8RecoJetsForBTagging = cms.InputTag("ak8PFJetsCHS")
    , mode = cms.string("JetLevel")
    
    , siPixelRecHitCollection = cms.InputTag('siPixelRecHits')
    , siStripRecHitCollection =  cms.VInputTag(
    cms.InputTag('siStripMatchedRecHits:rphiRecHit'),
    cms.InputTag('siStripMatchedRecHits:stereoRecHit'),
    cms.InputTag('siStripMatchedRecHits:rphiRecHitUnmatched'),
    cms.InputTag('siStripMatchedRecHits:stereoRecHitUnmatched')
    )
    , siStripMatchedRecHitCollection = cms.InputTag('siStripMatchedRecHits:matchedRecHit')

    # Jet level cfg
    , nJets = cms.int32(2)
    , minJetPt = cms.double(35.)
    , maxJetEta = cms.double(2.4)
    , z0PVCut  = cms.double(0.1)
    , isTTbar = cms.bool(True)
    , doECALstitched = cms.bool(True)
    , doTracksAtECALstitchedPt = cms.bool(True)
    , doTracksAtECALadjPt = cms.bool(True)
    , doHBHEenergy = cms.bool(True)
    , TauModelName = cms.string('tfModels/ResNet_8_channel_tf13.pb')
    )
