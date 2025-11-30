import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.register('skipEvents',
    default=0,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.int,
    info = "skipEvents")
options.parseArguments()

process = cms.Process("EGFrameProducer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = cms.string('102X_upgrade2018_realistic_v15')
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(options.maxEvents)
    input = cms.untracked.int32(10)
    )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      #options.inputFiles
      "file:SinglePhotonPt50_noPU_AODSIM.root"
      )
    , skipEvents = cms.untracked.uint32(0)#options.skipEvents
    )
print (" >> Loaded",len(options.inputFiles),"input files from list.")

#process.load("E2eDL.E2eDLrec.DetImg_cfi")
process.load("RecoE2E.FrameProducers.EGFrameProducer_cfi")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('SinglePhotonPt50_noPU_AODSIM+EGFrames.root')
    )
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("ntuple.root")#options.outputFile
    )

process.p = cms.Path(process.ProducerFrames+process.EGFrames)
process.ep=cms.EndPath(process.out)

#process.Timing = cms.Service("Timing",
#  summaryOnly = cms.untracked.bool(False),
#  useJobReport = cms.untracked.bool(True)
#)
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#    ignoreTotal = cms.untracked.int32(1)
#)