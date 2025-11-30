# RecoE2E
Read about the project and my work [here](https://medium.com/@purva.chaudhari02/google-summer-of-code-2021-5cf8ef45d2d2) 
<br>
#### Run the code
1. Set CMSSW envirnoment on docker/ lxplus:
   ```sh
   scram p <CMSSW version eg: CMSSW_10_6_20>
   cd CMSSW_10_6_20/src
   cmsenv
   ```
2. Git clone the repository:
   ```sh
   git clone -b taubranch4 https://github.com/Purva-Chaudhari/RecoE2E

   ```

3. Compile/Build. For using multi-core processor add -j n
   ```sh
   scram b -j 5
   ```
4. Run the inference (eg Tau Tagger). (Make sure you add the root files to your remote)
   ```sh
   cmsRun RecoE2E/TauTagger/python/TauInference_cfg.py inputFiles=file:./TTbar_TuneCUETP8M1_13TeV_pythia8_2018.root doTracksAtECALadjPt=False TauModelName=ResNet_8_channel_tf13.pb doBPIX3=False doBPIX4=False doTOB=False doTIB=False doTID=False
   ```
