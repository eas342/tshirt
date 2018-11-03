import phot_pipeline
import glob

yamlList = glob.glob('parameters/phot_param_2016*.yaml')

for oneYaml in yamlList:
    phot = phot_pipeline.phot(paramFile=oneYaml)
    phot.get_allimg_cen(recenter=True)
    


