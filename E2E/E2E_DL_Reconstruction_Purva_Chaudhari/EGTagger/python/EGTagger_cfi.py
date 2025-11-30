import FWCore.ParameterSet.Config as cms

EGTagger = cms.EDProducer('EGTagger'
    , photonCollection = cms.InputTag('gedPhotons')
    , EGFrames = cms.InputTag('EGFrames','EGFrames')
    , EGModelName = cms.string('tfModels/e_vs_ph_model.pb')
    )
