import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')

# TODO: put this option in cmsRun scripts
options.register('processMode', 
    default='JetLevel', 
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.string,
    info = "process mode: JetLevel or EventLevel")
# Set doEBenergy to 1 to produce EGSeeds and EGFrames.
options.register('doEBenergy',
    default=False,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doEBenergy")
# Skip Events.
options.register('skipEvents',
    default=0,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.int,
    info = "skipEvents")
# Name of the EGInference model to be used for inference.
options.register('EGModelName',
    default='e_vs_ph_model.pb',
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.string,
    info = "EGInference Model name")
options.parseArguments()

process = cms.Process("EGClassifier")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = cms.string('102X_upgrade2018_realistic_v15')
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
    #input = cms.untracked.int32(10)
    )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      options.inputFiles
      #"file:myOutputFile.root"#SinglePhotonPt50_noPU_AODSIM.root
      )
    , skipEvents = cms.untracked.uint32(0)#options.skipEvents
    )
print (" >> Loaded",len(options.inputFiles),"input files from list.")

process.load("RecoE2E.FrameProducers.DetFrameProducer_cfi")
process.load("RecoE2E.FrameProducers.EGFrameProducer_cfi")
process.load("RecoE2E.EGTagger.EGTagger_cfi")
#process.EGTagger.EGModelName = options.EGModelName
process.DetFrames.doHBHEenergy = False
process.DetFrames.doTracksAtECALstitchedPt = False
process.DetFrames.doBPIX1 = False
process.DetFrames.doBPIX2 = False
process.DetFrames.doBPIX3 = False
process.DetFrames.doBPIX4 = False
process.DetFrames.doTracksAtECALadjPt = False
process.DetFrames.setChannelOrder = "1"

process.EGTagger.EGmodelName = cms.string("tfModels/"+options.EGModelName)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('SinglePhotonPt50_noPU_AODSIM+EGFrames.root') 
    )
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("ntuple.root")#options.outputFile
    )

process.p = cms.Path(process.DetFrames + process.EGFrames+process.EGTagger)
process.ep=cms.EndPath(process.out)

#process.Timing = cms.Service("Timing",
#  summaryOnly = cms.untracked.bool(False),
#  useJobReport = cms.untracked.bool(True)
#)
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#    ignoreTotal = cms.untracked.int32(1)
#)
