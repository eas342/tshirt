import fit_tser_emcee

def prepDustMod166():
    customSpec= 'spectra/specific/hd166191/depth_vs_wavl.txt'
    mcObj = fit_tser_emcee.prepEmceeSpec(customSpec=customSpec,
                                         method='mie',
                                         customGuess=[30,0.7])
    return mcObj
    
def do_hd166():
    mcObj = prepDustMod166()
    mcObj.runMCMC()
    mcObj.showResult()
    mcObj.doCorner()
    print(mcObj.results)
    