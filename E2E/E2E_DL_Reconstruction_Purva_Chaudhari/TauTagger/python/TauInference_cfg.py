import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')

# TODO: put this option in cmsRun scripts
options.register('processMode', 
    default = 'JetLevel', 
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.string,
    info = "process mode: JetLevel or EventLevel")
# Skip Events.
options.register('skipEvents',
    default = 0,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.int,
    info = "skipEvents")
# Set doECALstitched to 1 to produce JetSeeds and JetFrames.
options.register('doECALstitched',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doECALstitched")
# Set doTracksAtECALstitchedPt to 1 to produce JetSeeds and JetFrames.
options.register('doTracksAtECALstitchedPt',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doTracksAtECALstitchedPt")
# Set doTracksAtECALadjPt to 1 to produce JetSeeds and JetFrames.
options.register('doTracksAtECALadjPt',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doTracksAtECALadjPt")
# Set doHBHEenergy to 1 to produce JetSeeds and JetFrames.
options.register('doHBHEenergy',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doHBHEenergy")
# Set doBPIX to 1 to producer BPIX layers
options.register('doBPIX1',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doBPIX1")
options.register('doBPIX2',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doBPIX2")
options.register('doBPIX3',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doBPIX3")
options.register('doBPIX4',
    default = True,
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.bool,
    info = "set doBPIX4")
# Set doTOB to 1 to producer TOB layers
options.register('doTOB',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doTOB")
# Set doTOB to 1 to producer TIB layers
options.register('doTIB',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doTIB")
# Set doTOB to 1 to producer TEC layers
options.register('doTEC',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doTEC")
# Set doTOB to 1 to producer TID layers
options.register('doTID',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doTID")
# Set order of the channels
options.register('setChannelOrder',
    default = "0,1,2,3,4,5,6,7",
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.string,
    info = "set the order of the channels")
# Name of the TauInference model to be used for inference.
options.register('TauModelName',
    default = 'ResNet_8_channel_tf13.pb',
    mult = VarParsing.VarParsing.multiplicity.singleton,
    mytype = VarParsing.VarParsing.varType.string,
    info = "TauInference Model name")
options.parseArguments()

process = cms.Process("TauClassifier")

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
process.load("RecoE2E.FrameProducers.JetFrameProducer_cfi")
process.load("RecoE2E.TauTagger.TauTagger_cfi")
#process.EGTagger.EGModelName = options.EGModelName
process.JetFrames.jetCollection = cms.string("ak8")
process.JetFrames.minJetPt = cms.double(35.)
process.JetFrames.maxJetEta = cms.double(2.4)
process.JetFrames.doHBHEenergy = options.doHBHEenergy
process.JetFrames.doECALstitched = options.doECALstitched
process.JetFrames.doTracksAtECALstitchedPt = options.doTracksAtECALstitchedPt
process.JetFrames.doTracksAtECALadjPt = options.doTracksAtECALadjPt
process.JetFrames.doBPIX1 = options.doBPIX1
process.JetFrames.doBPIX2 = options.doBPIX2
process.JetFrames.doBPIX3 = options.doBPIX3
process.JetFrames.doBPIX4 = options.doBPIX4
process.JetFrames.doTOB = options.doTOB
process.JetFrames.doTIB = options.doTIB
process.JetFrames.doTEC = options.doTEC
process.JetFrames.doTID = options.doTID

process.DetFrames.doHBHEenergy = options.doHBHEenergy
process.DetFrames.doECALstitched = options.doECALstitched
process.DetFrames.doTracksAtECALstitchedPt = options.doTracksAtECALstitchedPt
process.DetFrames.doTracksAtECALadjPt = options.doTracksAtECALadjPt
process.DetFrames.doBPIX1 = options.doBPIX1
process.DetFrames.doBPIX2 = options.doBPIX2
process.DetFrames.doBPIX3 = options.doBPIX3
process.DetFrames.doBPIX4 = options.doBPIX4
process.DetFrames.doTOB = options.doTOB
process.DetFrames.doTIB = options.doTIB
process.DetFrames.doTEC = options.doTEC
process.DetFrames.doTID = options.doTID
process.DetFrames.setChannelOrder = options.setChannelOrder

process.TauTagger.TauModelName = cms.string("tfModels/"+options.TauModelName)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TauPt+TauFrames.root') 
    )
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("ntuple.root")#options.outputFile
    )

process.p = cms.Path(process.DetFrames + process.JetFrames+process.TauTagger)
process.ep=cms.EndPath(process.out)

#process.Timing = cms.Service("Timing",
#  summaryOnly = cms.untracked.bool(False),
#  useJobReport = cms.untracked.bool(True)
#)
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#    ignoreTotal = cms.untracked.int32(1)
#)
