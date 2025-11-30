import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('skipEvents', 
    default=0, 
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.int,
    info = "skipEvents")
# TODO: put this option in cmsRun scripts
options.register('processMode', 
    default='JetLevel', 
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.string,
    info = "process mode: JetLevel or EventLevel")
# Set doEBenergy to 1 to produce EGSeeds and EGFrames.
#options.register('doEBenergy',
#    default=False,
#    mult=VarParsing.VarParsing.multiplicity.singleton,
#    mytype=VarParsing.VarParsing.varType.bool,
#    info = "set doEBenergy")
# Set doECALstitched to 1 to produce JetSeeds and JetFrames.
options.register('doECALstitched',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doECALstitched")
# Set doTracksAtECALstitchedPt to 1 to produce JetSeeds and JetFrames.
options.register('doTracksAtECALstitchedPt',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doTracksAtECALstitchedPt")
# Set doTracksAtECALadjPt to 1 to produce JetSeeds and JetFrames.
options.register('doTracksAtECALadjPt',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doTracksAtECALadjPt")
# Set doHBHEenergy to 1 to produce JetSeeds and JetFrames.
options.register('doHBHEenergy',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doHBHEenergy")
# Set doBPIX to 1 to producer BPIX layers
options.register('doBPIX1',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doBPIX1")
options.register('doBPIX2',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doBPIX2")
options.register('doBPIX3',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
    info = "set doBPIX3")
options.register('doBPIX4',
    default=True,
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.bool,
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
# Name of the jets to be used.
options.register('jetCollection',
    default='ak8',
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.string,
    info = "Jets: ak4/ak8")
# Name of the EGInference model to be used for inference.
options.register('EGModelName',
    default='e_vs_ph_model.pb',
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.string,
    info = "EGInference Model name")
# Name of the JetInference model to be used for inference.
options.register('JetsModelName',
    default='ResNet.pb',
    mult=VarParsing.VarParsing.multiplicity.singleton,
    mytype=VarParsing.VarParsing.varType.string,
    info = "Jets Inference Model name")
options.parseArguments()

process = cms.Process("Classifier")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
                                     destinations   = cms.untracked.vstring('detailedInfo'),
                                     detailedInfo   = cms.untracked.PSet(threshold  = cms.untracked.string('INFO'))
                                   )
                                    
process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi");
#process.load("Geometry.CaloEventSetup.CaloGeometry_cfi");
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi");
process.GlobalTag.globaltag = cms.string('80X_dataRun2_HLT_v12')
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(options.maxEvents) 
    #input = cms.untracked.int32(1000) 
    #input = cms.untracked.int32(-1) 
    #input = cms.untracked.int32(1000000) 
    )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      options.inputFiles
      )#"file:/afs/cern.ch/user/s/schaudha/public/CMSSW_10_6_8/src/demo/ZprimeToTT_M-2000_W-20_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_AODSIM_PUMoriond17.root"
    , skipEvents = cms.untracked.uint32(0)#options.skipEvents
    )
print (" >> Loaded",len(options.inputFiles),"input files from list.")

#process.load("ProdTutorial.ProducerTest.DetImg_cfi")
process.load("RecoE2E.FrameProducers.DetFrameProducer_cfi")
process.load("RecoE2E.FrameProducers.EGFrameProducer_cfi")
process.load("RecoE2E.FrameProducers.JetFrameProducer_cfi")

# Set JetCollection parameters for selected jet (ak8/ak4). 
process.JetFrames.jetCollection = options.jetCollection
if options.jetCollection == 'ak4':
    process.JetFrames.minJetPt = cms.double(35.)
    process.JetFrames.maxJetEta = cms.double(2.4)
elif options.jetCollection == 'ak8':
    process.JetFrames.minJetPt = cms.double(400.)
    process.JetFrames.maxJetEta = cms.double(1.37)
process.JetFrames.doHBHEenergy = options.doHBHEenergy
process.JetFrames.doECALstitched = options.doECALstitched
process.JetFrames.doTracksAtECALstitchedPt = options.doTracksAtECALstitchedPt
process.JetFrames.doTracksAtECALadjPt = options.doTracksAtECALadjPt
process.JetFrames.doBPIX1 = options.doBPIX1
process.JetFrames.doBPIX2 = options.doBPIX2
process.JetFrames.doBPIX3 = options.doBPIX3
process.JetFrames.doBPIX4 = options.doBPIX4

process.DetFrames.doHBHEenergy = options.doHBHEenergy
process.DetFrames.doECALstitched = options.doECALstitched
process.DetFrames.doTracksAtECALstitchedPt = options.doTracksAtECALstitchedPt
process.DetFrames.doTracksAtECALadjPt = options.doTracksAtECALadjPt
process.DetFrames.doBPIX1 = options.doBPIX1
process.DetFrames.doBPIX2 = options.doBPIX2
process.DetFrames.doBPIX3 = options.doBPIX3
process.DetFrames.doBPIX4 = options.doBPIX4


#process.out = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('myOutputFile.root')
#    ,outputCommands = cms.untracked.vstring('drop *',
#      "keep *_generalTracks_*_*",
#      "keep *_globalMuons_*_*",
#       "keep *_MuonTrackPoints_*_*",
#      "keep *_TrackTrackPoints_*_*")
#)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root'))
#print " >> Processing as:",(process.fevt_tf.mode)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("myoutput.root")#options.outputFile
   )

process.p = cms.Path(process.DetFrames + process.EGFrames + process.JetFrames)
process.ep=cms.EndPath(process.out)
process.Timing = cms.Service("Timing",
  summaryOnly = cms.untracked.bool(False),
  useJobReport = cms.untracked.bool(True)
)
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(1)
)
