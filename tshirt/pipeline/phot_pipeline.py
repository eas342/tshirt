import photutils
from astropy.io import fits, ascii
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from pkg_resources import resource_filename
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
import glob
from photutils import CircularAperture, CircularAnnulus
from photutils import RectangularAperture
from photutils import aperture_photometry
import photutils
if photutils.__version__ > "1.0":
    from . import fit_2dgauss
    from photutils.centroids import centroid_2dg
else:
    from photutils import centroid_2dg
import numpy as np
from astropy.time import Time
import astropy.units as u
import pdb
from copy import deepcopy
import yaml
import warnings
from scipy.stats import binned_statistic
from astropy.table import Table
import multiprocessing
from multiprocessing import Pool
import time
import logging
import urllib
import tqdm

maxCPUs = multiprocessing.cpu_count() // 3
try:
    import bokeh.plotting
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.models import Range1d
    from bokeh.models import WheelZoomTool
except ImportError as err2:
    print("Could not import bokeh plotting. Interactive plotting may not work")
from .utils import robust_poly, robust_statistics
from .utils import get_baseDir
from .instrument_specific import rowamp_sub

def run_one_phot_method(allInput):
    """
    Do a photometry/spectroscopy method on one file
    For example, do aperture photometry on one file
    This is a slightly awkward workaround because multiprocessing doesn't work on object methods
    So it's a separate function that takes an object and runs the method
    
    Parameters
    -----------
    allInput: 3 part tuple (object, int, string)
        This contains the object, file index to run (0-based) and name of the method to run
    """
    photObj, ind, method = allInput
    photMethod = getattr(photObj,method)
    return photMethod(ind)


def run_multiprocessing_phot(photObj,fileIndices,method='phot_for_one_file'):
    """
    Run photometry/spectroscopy methods on all files using multiprocessing
    Awkward workaround because multiprocessing doesn't work on object methods
    
    Parameters
    ----------
    photObj: Photometry object
        A photometry Object instance
    fileIndices: list
        List of file indices
    method: str
        Method on which to apply multiprocessing
    """
    allInput = []
    for oneInd in fileIndices:
        allInput.append([photObj,oneInd,method])
    
    n_files = len(fileIndices)
    if n_files < maxCPUs:
        raise Exception("Fewer files to process than CPUs, this can confuse multiprocessing")
    
    p = Pool(maxCPUs)
    outputDat = list(tqdm.tqdm(p.imap(run_one_phot_method,allInput),total=n_files))
    
    p.close()
    return outputDat

def read_yaml(filePath):
    with open(filePath) as yamlFile:
        yamlStructure = yaml.safe_load(yamlFile)
    return yamlStructure

path_to_example = "parameters/phot_params/example_phot_parameters.yaml"
exampleParamPath = resource_filename('tshirt',path_to_example)

class phot:
    def __init__(self,paramFile=exampleParamPath,
                 directParam=None):
        """ Photometry class
    
        Parameters
        ------
        paramFile: str
            Location of the YAML file that contains the photometry parameters as long
            as directParam is None. Otherwise, it uses directParam
        directParam: dict
            Rather than use the paramFile, you can put a dictionary here.
            This can be useful for running a batch of photometric extractions.
        
        Properties
        -------
        paramFile: str
            Same as paramFile above
        param: dict
            The photometry parameters like file names, aperture sizes, guess locations
        fileL: list
            The files on which photometry will be performed
        nImg: int
            Number of images in the sequence
        directParam: dict
            Parameter dictionary rather than YAML file (useful for batch processing)
        """
        self.pipeType = 'photometry'
        self.get_parameters(paramFile=paramFile,directParam=directParam)
        
        defaultParams = {'srcGeometry': 'Circular', 'bkgSub': True, 'isCube': False, 'cubePlane': 0,
                        'doCentering': True, 'bkgGeometry': 'CircularAnnulus',
                        'boxFindSize': 18,'backStart': 9, 'backEnd': 12,
                        'scaleAperture': False, 'apScale': 2.5, 'apRange': [0.01,9999],
                        'scaleBackground': False,
                        'nanTreatment': 'zero', 'backOffset': [0.0,0.0],
                        'srcName': 'WASP 62','srcNameShort': 'wasp62',
                         'refStarPos': [[50,50]],'procFiles': '*.fits',
                         'apRadius': 9,'FITSextension': 0,
                         'jdRef': 2458868,
                         'nightName': 'UT2020-01-20','srcName'
                         'FITSextension': 0, 'HEADextension': 0,
                         'refPhotCentering': None,'isSlope': False,
                         'itimeKeyword': 'INTTIME','readNoise': None,
                         'detectorGain': None,'cornerSubarray': False,
                         'subpixelMethod': 'exact','excludeList': None,
                         'dateFormat': 'Two Part','copyCentroidFile': None,
                         'bkgMethod': 'mean','diagnosticMode': False,
                         'bkgOrderX': 1, 'bkgOrderY': 1,'backsub_directions': ['Y','X'],
                         'readFromTshirtExamples': False,
                         'saturationVal': None, 'satNPix': 5, 'nanReplaceValue': 0.0,
                         'DATE-OBS': None,
                         'driftFile': None
                         }
        
        
        for oneKey in defaultParams.keys():
            if oneKey not in self.param:
                self.param[oneKey] = defaultParams[oneKey]
        
        xCoors, yCoors = [], []
        positions = self.param['refStarPos']
        self.nsrc = len(positions)
        
        ## Set up file names for output
        self.check_file_structure()
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        self.photFile = os.path.join(self.baseDir,'tser_data','phot','phot_'+self.dataFileDescrip+'.fits')
        self.centroidFile = os.path.join(self.baseDir,'centroids','cen_'+self.dataFileDescrip+'.fits')
        self.refCorPhotFile = os.path.join(self.baseDir,'tser_data','refcor_phot','refcor_'+self.dataFileDescrip+'.fits')
        
        # Get the file list
        self.fileL = self.get_fileList()
        self.nImg = len(self.fileL)
        
        self.srcNames = np.array(np.arange(self.nsrc),dtype=str)
        self.srcNames[0] = 'src'
        
        self.set_up_apertures(positions)
        
        self.check_parameters()
        self.get_drift_dat()
        
    
    def get_parameters(self,paramFile,directParam=None):
        if directParam is None:
            self.paramFile = paramFile
            self.param = read_yaml(paramFile)
        else:
            self.paramFile = 'direct dictionary'
            self.param = directParam
    
    def check_file_structure(self):
        """
        Check the file structure for plotting/saving data
        """
        baseDir = get_baseDir()
        structure_file = resource_filename('tshirt','directory_info/directory_list.yaml')
        dirList = read_yaml(structure_file)
        for oneFile in dirList:
            fullPath = os.path.join(baseDir,oneFile)
            ensure_directories_are_in_place(fullPath)
            
        self.baseDir = baseDir
    
    def get_fileList(self):
        if self.param['readFromTshirtExamples'] == True:
            ## Find the files from the package data examples
            ## This is only when running example pipeline runs or tests
            search_path = os.path.join(self.baseDir,'example_tshirt_data',self.param['procFiles'])
            if len(glob.glob(search_path)) == 0:
                print("Did not find example tshirt data. Now attempting to download...")
                get_tshirt_example_data()
        else:
            search_path = self.param['procFiles']
        
        origList = np.sort(glob.glob(search_path))
        if self.param['excludeList'] is not None:
            fileList = []
            
            for oneFile in origList:
                if os.path.basename(oneFile) not in self.param['excludeList']:
                    fileList.append(oneFile)
        else:
            fileList = origList
        
        if len(fileList) == 0:
            print("Note: File Search comes up empty")
            if os.path.exists(self.photFile):
                print("Note: Reading file list from previous phot file instead.")
                t1 = Table.read(self.photFile,hdu='FILENAMES')
                fileList = np.array(t1['File Path'])
        
        return fileList

    def check_parameters(self):
        assert type(self.param['backOffset']) == list,"Background offset is not a list"
        assert len(self.param['backOffset']) == 2,'Background offset must by a 2 element list'
    
    def set_up_apertures(self,positions):
        if self.param['srcGeometry'] == 'Circular':
            self.srcApertures = CircularAperture(positions,r=self.param['apRadius'])
        elif self.param['srcGeometry'] == 'Square':
            self.srcApertures = RectangularAperture(positions,w=self.param['apRadius'],
                                                    h=self.param['apRadius'],theta=0)
        elif self.param['srcGeometry'] == 'Rectangular':
            self.srcApertures = RectangularAperture(positions,w=self.param['apWidth'],
                                                    h=self.param['apHeight'],theta=0)
        else:
            print('Unrecognized aperture')
        
        self.xCoors = self.srcApertures.positions[:,0]
        self.yCoors = self.srcApertures.positions[:,1]
        
        if self.param['bkgSub'] == True:
            bkgPositions = np.array(deepcopy(positions))
            bkgPositions[:,0] = bkgPositions[:,0] + self.param['backOffset'][0]
            bkgPositions[:,1] = bkgPositions[:,1] + self.param['backOffset'][1]
            
            if self.param['bkgGeometry'] == 'CircularAnnulus':
                self.bkgApertures = CircularAnnulus(bkgPositions,r_in=self.param['backStart'],
                                                    r_out=self.param['backEnd'])
            elif self.param['bkgGeometry'] == 'Rectangular':
                self.bkgApertures = RectangularAperture(bkgPositions,w=self.param['backWidth'],
                                                    h=self.param['backHeight'],theta=0)
            else:
                raise ValueError('Unrecognized background geometry')
    
    def get_default_index(self):
        """
        Get the default index from the file list
        """
        return self.nImg // 2
    
    def get_default_im(self,img=None,head=None):
        """ Get the default image for postage stamps or star identification maps"""
        ## Get the data
        if img is None:
            img, head = self.getImg(self.fileL[self.get_default_index()])
        
        return img, head
    
    def get_default_cen(self,custPos=None,ind=0):
        """ 
        Get the default centroids for postage stamps or star identification maps
        
        Parameters
        ----------
        custPos: numpy array
            Array of custom positions for the apertures. Otherwise it uses the guess position
        ind: int
            Image index. This is used to guess the position if a drift file is given
        """
        if custPos is None:
            initialPos = deepcopy(self.srcApertures.positions)
            showApPos = np.zeros_like(initialPos)
            showApPos[:,0] = initialPos[:,0] + float(self.drift_dat['dx'][ind])
            showApPos[:,1] = initialPos[:,1] + float(self.drift_dat['dy'][ind])
            
        else:
            showApPos = custPos
        
        return showApPos
    
    def get_drift_dat(self):
        drift_dat_0 = Table()
        drift_dat_0['Index'] = np.arange(self.nImg)
        #drift_dat_0['File'] = self.fileL
        drift_dat_0['dx'] = np.zeros(self.nImg)
        drift_dat_0['dy'] = np.zeros(self.nImg)
        
        if self.param['driftFile'] == None:
            self.drift_dat = drift_dat_0
            drift_file_found = False
        else:
            if self.param['readFromTshirtExamples'] == True:
                ## Find the files from the package data examples
                ## This is only when running example pipeline runs or tests
                drift_file_path = os.path.join(self.baseDir,'example_tshirt_data',self.param['driftFile'])
            else:
                drift_file_path = self.param['driftFile']
            
            if os.path.exists(drift_file_path) == False:
                drift_file_found = False
                warnings.warn("No Drift file found at {}".format(drift_file_path))
            else:
                drift_file_found = True
                self.drift_dat = ascii.read(drift_file_path)
        return drift_file_found
    
    def make_drift_file(self,srcInd=0,refIndex=0):
        """
        Use the centroids in photometry to generate a drift file of X/Y offsets
        
        Parameters
        ----------
        srcInd: int
            The source index used for drifts
        refIndex: int
            Which file index corresponds to 0.0 drift
        """
        HDUList = fits.open(self.photFile)
        cenData = HDUList['CENTROIDS'].data
        photHead = HDUList['PHOTOMETRY'].header
        
        nImg = photHead['NIMG']
        drift_dat = Table()
        drift_dat['Index'] = np.arange(nImg)
        x = cenData[:,srcInd,0]
        drift_dat['dx'] = x - x[refIndex]
        y = cenData[:,srcInd,1]
        drift_dat['dy'] = y - y[refIndex]
        drift_dat['File'] = HDUList['FILENAMES'].data['File Path']
        outPath = os.path.join(self.baseDir,'centroids','drift_'+self.dataFileDescrip+'.ecsv')
        drift_dat.meta['Zero Index'] = refIndex
        drift_dat.meta['Source Used'] = srcInd
        drift_dat.meta['Zero File'] = str(drift_dat['File'][refIndex])
        print("Saving Drift file to {}".format(outPath))
        drift_dat.write(outPath,overwrite=True,format='ascii.ecsv')
    
    
    def showStarChoices(self,img=None,head=None,custPos=None,showAps=False,
                        srcLabel=None,figSize=None,showPlot=False,
                        apColor='black',backColor='black',
                        vmin=None,vmax=None,index=None,
                        labelColor='white',
                        xLim=None,yLim=None,
                        txtOffset=20):
        """
        Show the star choices for photometry
        
        Parameters
        ------------------
        img : numpy 2D array, optional
            An image to plot
        head : astropy FITS header, optional
            header for image
        custPos : numpy 2D array or list of tuple coordinates, optional
            Custom positions
        showAps : bool, optional
            Show apertures rather than circle stars
        srcLabel : str or None, optional
            What should the source label be? The default is "src"
        srcLabel : list or None, optional
            Specify the size of the plot.
            This is useful for looking at high/lower resolution
        showPlot : bool
            Show the plot? If True, it will show, otherwise it is saved as a file
        apColor: str
            The color for the source apertures
        backColor: str
            The color for the background apertures
        vmin: float or None
            A value for the :code:`matplotlib.pyplot.plot.imshow` vmin parameter
        vmax: float or None
            A value for the :code:`matplotlib.pyplot.plot.imshow` vmax parameter
        index: int or None
            The index of the file name. If None, it uses the default
        labelColor: str
            Color for the text label for sources
        xLim: None or two element list
            Specify the minimum and maximum X for the plot. For example xLim=[40,60]
        yLim: None or two element list
            Specify the minimum and maximum Y for the plot. For example yLim=[40,60]
        txtOffset: float
            The X and Y offset to place the text label for a source
        """
        fig, ax = plt.subplots(figsize=figSize)
        
        if index is None:
            index = self.get_default_index()
        
        if img is None:
            img, head = self.getImg(self.fileL[index])
        else:
            img_other, head = self.get_default_im(img=img,head=None)
        
        if vmin is None:
            useVmin = np.nanpercentile(img,1)
        else:
            useVmin = vmin
        
        if vmax is None:
            useVmax = np.nanpercentile(img,99)
        else:
            useVmax = vmax
        
        imData = ax.imshow(img,cmap='viridis',vmin=useVmin,vmax=useVmax,interpolation='nearest')
        ax.invert_yaxis()
        rad = 50 ## the radius for the matplotlib scatter to show source centers
        
        
        showApPos = self.get_default_cen(custPos=custPos,ind=index)
        if showAps == True:
            apsShow = deepcopy(self.srcApertures)
            apsShow.positions = showApPos
            
            
            self.adjust_apertures(index)
            
            if photutils.__version__ >= "0.7":
                apsShow.plot(axes=ax,color=apColor)
            else:
                apsShow.plot(ax=ax,color=apColor)
            if self.param['bkgSub'] == True:
                backApsShow = deepcopy(self.bkgApertures)
                backApsShow.positions = showApPos
                backApsShow.positions[:,0] = backApsShow.positions[:,0] + self.param['backOffset'][0]
                backApsShow.positions[:,1] = backApsShow.positions[:,1] + self.param['backOffset'][1]
            
                if photutils.__version__ >= "0.7":
                    backApsShow.plot(axes=ax,color=backColor)
                else:
                    backApsShow.plot(ax=ax,color=backColor)
            outName = 'ap_labels_{}.pdf'.format(self.dataFileDescrip)
            
        else:
            ax.scatter(showApPos[:,0],showApPos[:,1], s=rad, facecolors='none', edgecolors='r')
            outName = 'st_labels_{}.pdf'.format(self.dataFileDescrip)
        
        for ind, onePos in enumerate(showApPos):
            
            #circ = plt.Circle((onePos[0], onePos[1]), rad, color='r')
            #ax.add_patch(circ)
            if ind == 0:
                if srcLabel is None:
                    name='src'
                else:
                    name=srcLabel
            else:
                name=str(ind)
            ax.text(onePos[0]+txtOffset,onePos[1]+txtOffset,name,color=labelColor)
        
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(imData,label='Counts',cax=cax)
        
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)
        
        if showPlot == True:
            fig.show()
        else:
            outF = os.path.join(self.baseDir,'plots','photometry','star_labels',outName)
            fig.savefig(outF,
                        bbox_inches='tight')
            plt.close(fig)

    def showStamps(self,img=None,head=None,custPos=None,custFWHM=None,
                   vmin=None,vmax=None,showPlot=False,boxsize=None,index=None):
        """
        Shows the fixed apertures on the image with postage stamps surrounding sources
        
        Parameters
        -----------
        index: int
            Index of the file list. This is needed if scaling apertures
        """ 
        
        ##  Calculate approximately square numbers of X & Y positions in the grid
        numGridY = int(np.floor(np.sqrt(self.nsrc)))
        numGridX = int(np.ceil(float(self.nsrc) / float(numGridY)))
        fig, axArr = plt.subplots(numGridY, numGridX)
        
        img, head = self.get_default_im(img=img,head=head)
        
        if boxsize == None:
            boxsize = self.param['boxFindSize']
        
        showApPos = self.get_default_cen(custPos=custPos)
        
        if index is None:
            index = self.get_default_index()
        
        self.adjust_apertures(index)
        
        for ind, onePos in enumerate(showApPos):
            if self.nsrc == 1:
                ax = axArr
            else:
                ax = axArr.ravel()[ind]
            
            yStamp_proposed = np.array(onePos[1] + np.array([-1,1]) * boxsize,dtype=np.int)
            xStamp_proposed = np.array(onePos[0] + np.array([-1,1]) * boxsize,dtype=np.int)
            xStamp, yStamp = ensure_coordinates_are_within_bounds(xStamp_proposed,yStamp_proposed,img)
            
            stamp = img[yStamp[0]:yStamp[1],xStamp[0]:xStamp[1]]
            
            if vmin == None:
                useVmin = np.nanpercentile(stamp,1)
            else:
                useVmin = vmin
            
            if vmax == None:
                useVmax = np.nanpercentile(stamp,99)
            else:
                useVmax = vmax
            
            
            imData = ax.imshow(stamp,cmap='viridis',vmin=useVmin,vmax=useVmax,interpolation='nearest')
            ax.invert_yaxis()
            
            ax.set_title(self.srcNames[ind])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            srcX, srcY = onePos[0] - xStamp[0],onePos[1] - yStamp[0]
            
            circ = plt.Circle((srcX,srcY),
                              self.srcApertures.r,edgecolor='red',facecolor='none')
            ax.add_patch(circ)
            if self.param['bkgSub'] == True:
                for oneRad in [self.bkgApertures.r_in, self.bkgApertures.r_out]:
                    circ = plt.Circle((srcX + self.param['backOffset'][0],srcY + self.param['backOffset'][1]),
                                      oneRad,edgecolor='blue',facecolor='none')
                    ax.add_patch(circ)
            
            if custFWHM is not None:
                circFWHM = plt.Circle((srcX,srcY),
                                      custFWHM[ind],edgecolor='orange',facecolor='none')
                ax.add_patch(circFWHM)
            
        
        fig.colorbar(imData,label='Counts')
        
        totStamps = numGridY * numGridX
        
        for ind in np.arange(self.nsrc,totStamps):
            ## Make any extra postage stamps blank
            ax = axArr.ravel()[ind]
            
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #self.srcApertures.plot(indices=ind,color='red')
            #ax.set_xlim(onePos[0] - boxsize,onePos[0] + boxsize)
            #ax.set_ylim(onePos[1] - boxsize,onePos[1] + boxsize)
        
        if showPlot == True:
            fig.show()
        else:
            outName = 'stamps_'+self.dataFileDescrip+'.pdf'
            outPath = os.path.join(self.baseDir,'plots','photometry','postage_stamps',outName)
            fig.savefig(outPath)
            plt.close(fig)
        
    def showCustSet(self,index=None,ptype='Stamps',defaultCen=False,vmin=None,vmax=None,
                    boxsize=None,showPlot=False):
        """ Show a custom stamp or star identification plot for a given image index 
        
        Parameters
        --------------
        index: int
            Index of the image/centroid to show
        ptype: str
            Plot type - 'Stamps' for postage stamps
                        'Map' for star identification map
        defaultCen: bool
            Use the default centroid? If True, it will use the guess centroids to 
                    show the guess before centering.
        boxsize: int or None
            The size of the box to cut out for plotting postage stamps.
            If None, will use defaults.
            Only use when ptype is 'Stamps'
        showPlot: bool
            Show the plot in notebook or iPython session?
            If True, it will show the plot.
            If False, it will save the plot in the default directory.
        """
        self.get_allimg_cen()
        if index == None:
            index = self.get_default_index()
        
        img, head = self.getImg(self.fileL[index])
        
        if defaultCen == True:
            cen = self.srcApertures.positions
        else:
            cen = self.cenArr[index]
        
        if ptype == 'Stamps':
            self.showStamps(custPos=cen,img=img,head=head,vmin=vmin,vmax=vmax,showPlot=showPlot,
                            boxsize=boxsize,index=index)
        elif ptype == 'Map':
            self.showStarChoices(custPos=cen,img=img,head=head,showAps=True,showPlot=showPlot,
                            index=index)
        else:
            print('Unrecognized plot type')
            
    
    def make_filename_hdu(self,airmass=None):
        """
        Makes a Header data unit (binary FITS table) for filenames
        """
        fileLTable = Table()
        fileLTable['File Path'] = self.fileL
        if airmass is not None:
            fileLTable['Airmass'] = airmass
        
        hduFileNames = fits.BinTableHDU(fileLTable)
        hduFileNames.name = "FILENAMES"
        
        return hduFileNames
    
    def save_centroids(self,cenArr,fwhmArr,fixesApplied=True,origCen=None,origFWHM=None):
        """ Saves the image centroid data
        Parameters
        -----------
        cenArr: numpy array
            3d array of centroids (nImg x nsrc x 2 for x/y)
        fwhmArr: numpy array
            3d array of fwhm (nImg x nsrc x 2 for x/y)
        fixesApplied: bool
            Are fixes applied to the centroids?
        origCen: None or numpy array
            Original array of centroids
        origFWHM: None or numpy array
            Original array of FWHMs
        """
        hdu = fits.PrimaryHDU(cenArr)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with centroids')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('dimension','dimension axis X=0,Y=1')
        hdu.header['AXIS2'] = ('src','source axis')
        hdu.header['AXIS3'] = ('image','image axis')
        hdu.header['BOXSZ'] = (self.param['boxFindSize'],'half-width of the box used for source centroids')
        hdu.header['REFCENS'] = (self.param['refPhotCentering'],'Reference Photometry file used to shift the centroids (or empty if none)')
        hdu.header['FIXES'] = (fixesApplied, 'Have centroid fixes been applied from trends in other sources?')
        hdu.name = 'Centroids'
        
        hdu2 = fits.ImageHDU(fwhmArr)
        hdu2.header['NSOURCE'] = (self.nsrc,'Number of sources with centroids')
        hdu2.header['NIMG'] = (self.nImg,'Number of images')
        hdu2.header['AXIS1'] = ('dimension','dimension axis X=0,Y=1')
        hdu2.header['AXIS2'] = ('src','source axis')
        hdu2.header['AXIS3'] = ('image','image axis')
        hdu2.header['BOXSZ'] = (self.param['boxFindSize'],'half-width of the box used to fit 2D gaussian')
        hdu2.name = 'FWHM'
        
        hduFileNames = self.make_filename_hdu()
        
        HDUList = fits.HDUList([hdu,hdu2,hduFileNames])
        
        if fixesApplied == True:
            if origCen is not None:
                hduCenOrig = fits.ImageHDU(origCen,hdu.header)
                hduCenOrig.header['FIXES'] = ('Unknown','Have centroid fixes been applied from trends in other sources?')
                hduCenOrig.name = 'ORIG CEN'
                HDUList.append(hduCenOrig)
            if origFWHM is not None:
                hduFWHMOrig = fits.ImageHDU(origFWHM,hdu2.header)
                hduFWHMOrig.name = 'ORIG FWHM'
                HDUList.append(hduFWHMOrig)
        
        HDUList.writeto(self.centroidFile,overwrite=True)
        
        head = hdu.header
        return head, hdu2.header
    
    def shift_centroids_from_other_file(self,refPhotFile,SWLW=True):
        """
        Creates a centroid array where shifts are applied from another 
        file.
        For example, Imaging data from another camera can be used
        to provide shifts to the apertures for grism data
        """
        if SWLW == True:
            rotAngle = 0; ## set to zero for now
            scaling = 0.5
        else:
            raise Exception("Still working on scaling and rotation params")
        
        HDUList = fits.open(refPhotFile)
        if "CENTROIDS" not in HDUList:
            raise Exception("Could not find CENTROIDS extension in {}".format(refPhotFile))
        
        refCenArr, head = HDUList["CENTROIDS"].data, HDUList["CENTROIDS"].header
        
        xRefAbs, yRefAbs = refCenArr[:,0,0], refCenArr[:,0,1]
        xRef, yRef = xRefAbs - xRefAbs[0], yRefAbs - yRefAbs[0]
        
        HDUList.close()
        
        ndim=2 ## Number of dimensions in image (assuming 2D)        
        cenArr = np.zeros((self.nImg,self.nsrc,ndim))
        
        pos = self.get_default_cen()
        for ind, oneFile in enumerate(self.fileL):
            xVec = (xRef[ind] * np.cos(rotAngle) - yRef[ind] * np.sin(rotAngle)) * scaling
            yVec = (xRef[ind] * np.sin(rotAngle) + yRef[ind] * np.cos(rotAngle)) * scaling
            cenArr[ind,:,0] = pos[:,0] + xVec
            cenArr[ind,:,1] = pos[:,1] + yVec
        fwhmArr = np.zeros_like(cenArr)
        return cenArr, fwhmArr
    
    def fix_centroids(self,diagnostics=False,nsigma=10.):
        """
        Fix the centroids for outlier positions for stars
        """
        HDUList = fits.open(self.centroidFile)
        cenArr, head = HDUList["CENTROIDS"].data, HDUList["CENTROIDS"].header
        fwhmArr, headFWHM = HDUList["FWHM"].data, HDUList["FWHM"].header
        
        fixedCenArr = deepcopy(cenArr)
        fixedFWHMArr = deepcopy(fwhmArr)
        
        medCen = np.nanmedian(cenArr,0)
        medCen3D = np.tile(medCen,[self.nImg,1,1])
        
        diffCen = cenArr - medCen3D ## differential motion
        fixedDiffCen = deepcopy(diffCen)
        
        if diagnostics == True:
            fig, axArr = plt.subplots(2,sharex=True)
        
        for oneAxis in [0,1]:
            trend = np.nanmedian(diffCen[:,:,oneAxis],1) ## median trend
            trend2D = np.tile(trend,[self.nsrc,1]).transpose()
            diffFromTrend = diffCen[:,:,oneAxis] - trend2D
            mad = np.nanmedian(np.abs(diffFromTrend))
            
            badPt = np.abs(diffFromTrend) > nsigma * mad
            fixedDiffFromTrend = deepcopy(diffFromTrend)
            fixedDiffFromTrend[badPt] = 0
            fwhmArr2D = fixedFWHMArr[:,:,oneAxis]
            fwhmArr2D[badPt] = np.nan ## w/ different position FWHM is no longer relvant
            fixedFWHMArr[:,:,oneAxis] = fwhmArr2D
            
            fixedDiffCen[:,:,oneAxis] = fixedDiffFromTrend + trend2D
            
            if diagnostics == True:
                for oneSrc in np.arange(self.nsrc):
                    showData, = axArr[oneAxis].plot(diffCen[:,oneSrc,oneAxis],'o')
                    axArr[oneAxis].plot(fixedDiffCen[:,oneSrc,oneAxis],color=showData.get_color())
        
        fixedCenArr = fixedDiffCen + medCen3D
        
        if diagnostics == True:
            plt.show()
        
        self.save_centroids(fixedCenArr,fixedFWHMArr,fixesApplied=True,origCen=cenArr,origFWHM=fwhmArr)
        
        HDUList.close()
    
    def copy_centroids_from_file(self,fileName):
        HDUList = fits.open(fileName)
        cenArr, head = HDUList["CENTROIDS"].data, HDUList["CENTROIDS"].header
        if "FWHM" in HDUList:
            fwhmArr, headFWHM = HDUList["FWHM"].data, HDUList["FWHM"].header
            self.keepFWHM = True
        else:
            self.keepFWHM = False ## allow for legacy centroid files
            fwhmArr, headFWHM = None, None
        
        HDUList.close()
        return cenArr, head, fwhmArr, headFWHM
    
    def get_allimg_cen(self,recenter=False,useMultiprocessing=False):
        """ Get all image centroids
        If self.param['doCentering'] is False, it will just use the input aperture positions 
        Parameters
        ----------
        recenter: bool
            Recenter the apertures again? Especially after changing the sources in photometry parameters
        useMultiprocessing: bool
            Use multiprocessing for faster computation?s
        """
        
        ndim=2 ## Number of dimensions in image (assuming 2D)
        
        if os.path.exists(self.centroidFile) and (recenter == False):
            cenArr, head, fwhmArr, headFWHM = self.copy_centroids_from_file(self.centroidFile)
        elif (self.param['copyCentroidFile'] is not None) and (recenter == False):
            cenArr, head, fwhmArr, headFWHM = self.copy_centroids_from_file(self.param['copyCentroidFile'])
        elif self.param['refPhotCentering'] is not None:
            cenArr, fwhmArr = self.shift_centroids_from_other_file(self.param['refPhotCentering'])
            head, headFWHM = self.save_centroids(cenArr,fwhmArr)
            self.keepFWHM = False
            
        elif self.param['doCentering'] == False:
            img, head = self.get_default_im()
            cenArr = np.zeros((self.nImg,self.nsrc,ndim))
            pos = self.get_default_cen()
            for ind, oneFile in enumerate(self.fileL):
                cenArr[ind,:,0] = pos[:,0]
                cenArr[ind,:,1] = pos[:,1]
            fwhmArr = np.zeros_like(cenArr)
            head, headFWHM = self.save_centroids(cenArr,fwhmArr)
            self.keepFWHM = False
        else:
            cenArr = np.zeros((self.nImg,self.nsrc,ndim))
            fwhmArr = np.zeros((self.nImg,self.nsrc,ndim))
            
            if useMultiprocessing == True:
                fileCountArray = np.arange(len(self.fileL))
                allOutput = run_multiprocessing_phot(self,fileCountArray,method='get_allcen_img')
            else:
                allOutput = []
                for ind, oneFile in enumerate(self.fileL):
                    allOutput.append(self.get_allcen_img(ind))
            
            for ind, oneFile in enumerate(self.fileL):
                allX, allY, allfwhmX, allfwhmY = allOutput[ind]
                cenArr[ind,:,0] = allX
                cenArr[ind,:,1] = allY
                fwhmArr[ind,:,0] = allfwhmX
                fwhmArr[ind,:,1] = allfwhmY
            
            self.keepFWHM = True
                
            head, headFWHM = self.save_centroids(cenArr,fwhmArr)
            
        self.cenArr = cenArr
        self.cenHead = head
        
        if self.param['bkgSub'] == True:
            ## Make an array for the background offsets
            backgOffsetArr = np.zeros((self.nImg,self.nsrc,ndim))
            backgOffsetArr[:,:,0] = self.param['backOffset'][0]
            backgOffsetArr[:,:,1] = self.param['backOffset'][1]
            self.backgOffsetArr = backgOffsetArr
        else:
            self.backgOffsetArr = np.zeros((self.nImg,self.nsrc,ndim))
        
        if self.keepFWHM == True:
            self.fwhmArr = fwhmArr
            self.headFWHM = headFWHM
    
    def get_allcen_img(self,ind,showStamp=False):
        """ Gets the centroids for all sources in one image """
        img, head = self.getImg(self.fileL[ind])
        
        allX, allY = [], []
        allfwhmX, allfwhmY = [], []
        
        positions = self.get_default_cen(ind=ind)
        
        for srcInd, onePos in enumerate(positions):
            xcen, ycen, fwhmX, fwhmY = self.get_centroid(img,onePos[0],onePos[1])
            allX.append(xcen)
            allY.append(ycen)
            allfwhmX.append(fwhmX)
            allfwhmY.append(fwhmY)
        
        if showStamp == True:
            posArr = np.vstack((allX,allY)).transpose()
            #fwhmArr = np.vstack((allfwhmX,allfwhmY)).transpose()
            quadFWHM = np.sqrt(np.array(allfwhmX)**2 + np.array(allfwhmY)**2)
            self.showStamps(img=img,custPos=posArr,custFWHM=quadFWHM)
        return allX, allY, allfwhmX, allfwhmY
    
    def get_centroid(self,img,xGuess,yGuess):
        """ Get the centroid of a source given an x and y guess 
        Takes the self.param['boxFindSize'] to define the search box
        """
        boxSize=self.param['boxFindSize']
        shape = img.shape
        minX = int(np.max([xGuess - boxSize,0.]))
        maxX = int(np.min([xGuess + boxSize,shape[1]-1]))
        minY = int(np.max([yGuess - boxSize,0.]))
        maxY = int(np.min([yGuess + boxSize,shape[1]-1]))
        subimg = img[minY:maxY,minX:maxX]
        
        try:
            xcenSub,ycenSub = centroid_2dg(subimg)
        except ValueError:
            warnings.warn("Found value error for centroid. Putting in Guess value")
            xcenSub,ycenSub = xGuess, yGuess
        
        
        xcen = xcenSub + minX
        ycen = ycenSub + minY
        
        try:
            fwhmX, fwhmY = self.get_fwhm(subimg,xcenSub,ycenSub)
        except ValueError:
            warnings.warn("Found value error for FWHM. Putting in a Nan")
            fwhmX, fwhmY = np.nan, np.nan

        return xcen, ycen, fwhmX, fwhmY
    
    def get_fwhm(self,subimg,xCen,yCen):
        """ Get the FWHM of the source in a subarray surrounding it 
        """
        if photutils.__version__ >= "1.0":
            GaussFit = fit_2dgauss.centroid_2dg_w_sigmas(subimg)
            x_stddev = GaussFit[2]
            y_stddev = GaussFit[3]
        else:
            GaussModel = photutils.fit_2dgaussian(subimg)
            x_stddev = GaussModel.x_stddev.value
            y_stddev = GaussModel.y_stddev.value
        
        fwhmX = x_stddev * 2.35482005
        fwhmY = y_stddev * 2.35482005
        return fwhmX, fwhmY
    
    def add_filenames_to_header(self,hdu):
        """ Uses fits header cards to list the files
        This clutters up the header, so I have now moved
        the fileName list to a separate structure
        """
        for ind, oneFile in enumerate(self.fileL):
            hdu.header['FIL'+str(ind)] = (os.path.basename(oneFile),'file name')
        
    
    def reset_phot(self):
        """ 
        Reset the photometry
        
        A reminder to myself to write a script to clear the positions.
        Sometimes, if you get bad positions from a previous file, they 
        will wind up being used again. Need to reset the srcAperture.positions!
        """
    
    def get_date(self,head):
        if 'DATE-OBS' in head:
            useDate = head['DATE-OBS']
        elif 'DATE_OBS' in head:
            useDate = head['DATE_OBS']
        elif 'DATE' in head:
            warnings.warn('DATE-OBS not found in header. Using DATE instead')
            month1, day1, year1 = head['DATE'].split("/")
            useDate = "-".join([year1,month1,day1])
        elif 'DATE-OBS' in self.param:
            warnings.warn('Using DATE-OBS from parameter file.')
            useDate = self.param['DATE-OBS']
        else:
            warnings.warn('Date headers not found in header. Making it nan')
            useDate = np.nan
        
        if self.param['dateFormat'] == 'Two Part':
            t0 = Time(useDate+'T'+head['TIME-OBS'])
        elif self.param['dateFormat'] == 'One Part':
            t0 = Time(useDate)
        else:
            raise Exception("Date format {} not understdood".format(self.param['dateFormat']))
        
        
        if 'timingMethod' in self.param:
            if self.param['timingMethod'] == 'JWSTint':
                if 'INTTIME' in head:
                    int_time = head['INTTIME']
                elif 'EFFINTTM' in head:
                    int_time = head['EFFINTTM']
                else:
                    warnings.warn("Couldn't find inttime in header. Setting to 0")
                    int_time = 0
                
                t0 = t0 + (head['TFRAME'] + int_time) * (head['ON_NINT']) * u.second
            elif self.param['timingMethod'] == 'intCounter':
                t0 = t0 + (head['ON_NINT']) * 1.0 * u.min ## placeholder to spread out time
        
        return t0
    
    def get_read_noise(self,head):
        if self.param['readNoise'] != None:
            readNoise = float(self.param['readNoise'])
        elif 'RDNOISE1' in head:
            readNoise = float(head['RDNOISE1'])
        else:
            readNoise = 1.0
            warnings.warn('Warning, no read noise specified')
        
        return readNoise
    
    def adjust_apertures(self,ind):
        """
        Adjust apertures, if scaling by FWHM
        
        Parameters
        ----------
        ind: int
            the index of `self.fileList`.
        
        """
        if self.param['scaleAperture'] == True:
            if self.nImg >= maxCPUs:
                useMultiprocessing = True
            else:
                useMultiprocessing = False
            
            self.get_allimg_cen(useMultiprocessing=useMultiprocessing)
            
            medianFWHM = np.median(self.fwhmArr[ind])
        
            minFWHMallowed, maxFWHMallowed = self.param['apRange']
            
            if medianFWHM < minFWHMallowed:
                warnings.warn("FWHM found was smaller than apRange ({}) px. Using {} for Image {}".format(minFWHMallowed,minFWHMallowed,self.fileL[ind]))
                medianFWHM = minFWHMallowed
            elif medianFWHM > maxFWHMallowed:
                warnings.warn("FWHM found was larger than apRange ({}) px. Using {} for Image {}".format(maxFWHMallowed,maxFWHMallowed,self.fileL[ind]))
                medianFWHM = maxFWHMallowed
            
            if self.param['bkgGeometry'] == 'CircularAnnulus':
                self.srcApertures.r = medianFWHM * self.param['apScale']
                if self.param['scaleBackground'] == True:
                    self.bkgApertures.r_in = (self.srcApertures.r + 
                                              self.param['backStart'] - self.param['apRadius'])
                    self.bkgApertures.r_out = (self.bkgApertures.r_in +
                                               self.param['backEnd'] - self.param['backStart'])
            else:
                warnings.warn('Background Aperture scaling not set up for non-annular geometry')
    
    def get_ap_area(self,aperture):
        """
        A function go get the area of apertures
        This accommodates different versions of photutils
        """
        if photutils.__version__ > '0.6':
            ## photutils changed area from a method to an attribute after this version
            area = aperture.area
        else:
            ## older photutils
            area = aperture.area()
        return area
    
    def phot_for_one_file(self,ind):
        """
        Calculate aperture photometry using `photutils`
        
        Parameters
        ----------
        ind: int
            index of the file list on which to read in and do photometry
        """
        
        oneImg = self.fileL[ind]
        img, head = self.getImg(oneImg)
        
        t0 = self.get_date(head)
        
        self.srcApertures.positions = self.cenArr[ind]
        
        if self.param['scaleAperture'] == True:
            self.adjust_apertures(ind)
        
        readNoise = self.get_read_noise(head)
        
        err = np.sqrt(np.abs(img) + readNoise**2) ## Should already be gain-corrected
        
        rawPhot = aperture_photometry(img,self.srcApertures,error=err,method=self.param['subpixelMethod'])
        
        if self.param['saturationVal'] != None:
            src_masks = self.srcApertures.to_mask(method='center')
            for srcInd,mask in enumerate(src_masks):
                src_data = mask.multiply(img)
                src_data_1d = src_data[mask.data > 0]
                satPoints = (src_data_1d > self.param['saturationVal'])
                if np.sum(satPoints) >= self.param['satNPix']:
                    ## set this source as NaN b/c it's saturated
                    rawPhot['aperture_sum'][srcInd] = np.nan
        
        
        if self.param['bkgSub'] == True:
            self.bkgApertures.positions = self.cenArr[ind] + self.backgOffsetArr[ind]
            
            if self.param['bkgMethod'] == 'mean':
                bkgPhot = aperture_photometry(img,self.bkgApertures,error=err,method=self.param['subpixelMethod'])
                bkgArea = self.get_ap_area(self.bkgApertures)
                srcArea = self.get_ap_area(self.srcApertures)
                
                
                bkgVals = bkgPhot['aperture_sum'] / bkgArea * srcArea
                bkgValsErr = bkgPhot['aperture_sum_err'] / bkgArea * srcArea
                
                ## Background subtracted fluxes
                srcPhot = rawPhot['aperture_sum'] - bkgVals
            elif self.param['bkgMethod'] in ['median', 'robust mean']:
                bkgIntensity, bkgIntensityErr = [], []
                bkg_masks = self.bkgApertures.to_mask(method='center')
                for mask in bkg_masks:
                    bkg_data = mask.multiply(img)
                    bkg_data_1d = bkg_data[mask.data > 0]
                    oneIntensity, oneErr = robust_statistics(bkg_data_1d,method=self.param['bkgMethod'])
                    bkgIntensity.append(oneIntensity)
                    bkgIntensityErr.append(oneErr)
                
                bkgVals = np.array(bkgIntensity)  * self.get_ap_area(self.srcApertures)
                bkgValsErr = np.array(bkgIntensityErr) * self.get_ap_area(self.srcApertures)
            
                srcPhot = rawPhot['aperture_sum'] - bkgVals
            elif self.param['bkgMethod'] == 'colrow':
                srcPhot, bkgVals, bkgValsErr = self.poly_sub_phot(img,head,err,ind)
            elif self.param['bkgMethod'] == 'rowAmp':
                srcPhot, bkgVals, bkgValsErr = self.rowamp_sub_phot(img,head,err,ind)
            else:
                raise Exception("Unrecognized background method {}".format(self.param['bkgMethod']))
        else:
            ## No background subtraction
            srcPhot = rawPhot['aperture_sum']
            bkgVals = np.nan
            bkgValsErr = 0.
        
        srcPhotErr = np.sqrt(rawPhot['aperture_sum_err']**2 + bkgValsErr**2)
        
        
        return [t0.jd,srcPhot,srcPhotErr, bkgVals]
    
    def show_cutout(self,img,aps=None,name='',percentScaling=False,src=None,ind=None):
        """ Plot the cutout around the source for diagnostic purposes"""
        fig, ax = plt.subplots()
        if percentScaling == True:
            vmin,vmax = np.nanpercentile(img,[3,97])
        else:
            vmin,vmax = None, None
        
        ax.imshow(img,vmin=vmin,vmax=vmax)
        if aps is not None:
            if photutils.__version__ >= "0.7":
                aps.plot(axes=ax)
            else:
                aps.plot(ax=ax)
            
        ax.set_title(name)
        fig.show()
        print('Press c and enter to continue')
        print('or press q and enter to quit')
        pdb.set_trace()
        plt.close(fig)
        
        primHDU = fits.PrimaryHDU(img)
        primHDU.header['Name'] = name
        name_for_file = name.replace(" ","_")
        outName = "{}_{}_src_{}_ind_{}.fits".format(self.dataFileDescrip,name_for_file,src,ind)
        outPath = os.path.join("diagnostics","phot_poly_backsub",outName)
        primHDU.writeto(outPath,overwrite=True)
        
    
    def poly_sub_phot(self,img,head,err,ind,showEach=False,saveFits=False):
        """
        Do a polynomial background subtraction use robust polynomials
        
        This is instead of using the mean or another statistic of the background aperture
        """
        from . import spec_pipeline
        
        bkg_masks = self.bkgApertures.to_mask(method='center')
        
        spec = spec_pipeline.spec()
        spec.param['bkgOrderX'] = self.param['bkgOrderX']
        spec.param['bkgOrderY'] = self.param['bkgOrderY']
        spec.fileL = self.fileL
        
        srcPhot, bkgPhot = [], []
        for ind,mask in enumerate(bkg_masks):
            backImg = mask.multiply(img,fill_value=np.nan)
            ## fill value doesn't appear to work so I manually will make them NaN
            nonBackPts = mask.data == 0
            backImg[nonBackPts] = np.nan
            
            img_cutout = mask.cutout(img,fill_value=0.0)
            err_cutout = mask.cutout(err,fill_value=0.0)
            
            srcApSub = deepcopy(self.srcApertures)
            srcApSub.positions[:,0] = srcApSub.positions[:,0] - mask.bbox.ixmin
            srcApSub.positions[:,1] = srcApSub.positions[:,1] - mask.bbox.iymin
            
            spec.param['bkgRegionsX'] = [[0,backImg.shape[1]]]
            spec.param['bkgRegionsY'] = [[0,backImg.shape[0]]]
            
            if self.param['diagnosticMode'] == True:
                self.show_cutout(img_cutout,aps=srcApSub,name='Img Cutout',src=ind,ind=ind)
                self.show_cutout(backImg,aps=srcApSub,name='Background Cutout',
                                 percentScaling=True,src=ind,ind=ind)
            
            backImg_sub, bkgModelTotal, subHead = spec.do_backsub(backImg,head,ind=ind,
                                                                  directions=self.param['backsub_directions'])
            subImg = img_cutout - bkgModelTotal
            
            srcPhot1 = aperture_photometry(subImg,srcApSub,error=err_cutout,
                                          method=self.param['subpixelMethod'])
            srcPhot.append(srcPhot1['aperture_sum'][ind])
            bkgPhot1 = aperture_photometry(bkgModelTotal,srcApSub,error=err_cutout,
                                          method=self.param['subpixelMethod'])
            bkgPhot.append(bkgPhot1['aperture_sum'][ind])
            
            if self.param['diagnosticMode'] == True:
                self.show_cutout(subImg,aps=srcApSub,name='Backsub Img Cutout',
                                 percentScaling=True,src=ind,ind=ind)
        
        
        ## use the error in the mean background as an estimate for error
        bkgPhotTotal = aperture_photometry(img,self.bkgApertures,error=err,method=self.param['subpixelMethod'])
        bkgValsErr = (bkgPhotTotal['aperture_sum_err'] / self.get_ap_area(self.bkgApertures)
                     * self.get_ap_area(self.srcApertures))
        
        
        return np.array(srcPhot),np.array(bkgPhot),bkgValsErr
    
    def rowamp_sub_phot(self,img,head,err,ind):
        """
        Do a row-by-row, amplifier-by-amplifier background subtraction
        
        This is instead of using the mean or another statistic of the background aperture
        """
        
        saveD = self.param['diagnosticMode']
        backsub_img, backg_img = rowamp_sub.do_backsub(img,self,
                                                       saveDiagnostics=saveD)
        
        srcPhot_t = aperture_photometry(backsub_img,
                                        self.srcApertures,error=err,
                                        method=self.param['subpixelMethod'])
        
        bkgPhot_t = aperture_photometry(backg_img,
                                        self.srcApertures,error=err,
                                        method=self.param['subpixelMethod'])
        srcPhot = srcPhot_t['aperture_sum']
        bkgPhot = bkgPhot_t['aperture_sum']
        bkgValsErr = bkgPhot_t['aperture_sum_err']
        
        return np.array(srcPhot),np.array(bkgPhot),np.array(bkgValsErr)
    
    def return_self(self):
        return self
    
    def do_phot(self,useMultiprocessing=False):
        """ Does photometry using the centroids found in get_allimg_cen 
        """
        self.get_allimg_cen(useMultiprocessing=useMultiprocessing)
        
        photArr = np.zeros((self.nImg,self.nsrc))
        errArr = np.zeros_like(photArr)
        backArr = np.zeros_like(photArr)
        
        jdArr = []
        
        fileCountArray = np.arange(len(self.fileL))
        
        if useMultiprocessing == True:
            outputPhot = run_multiprocessing_phot(self,fileCountArray)
        else:
            outputPhot = []
            for ind in tqdm.tqdm(fileCountArray):
                outputPhot.append(self.phot_for_one_file(ind))
        
        ## unpack the results
        for ind,val in enumerate(outputPhot):
            jdArr.append(val[0])
            photArr[ind,:] = val[1]
            errArr[ind,:] = val[2]
            backArr[ind,:] = val[3]
            
        ## Save the photometry results
        hdu = fits.PrimaryHDU(photArr)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with photometry')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('src','source axis')
        hdu.header['AXIS2'] = ('image','image axis')
        basicHeader = deepcopy(hdu.header)
        
#        hdu.header[''] = '        Source parameters '
        hdu.header['SRCNAME'] = (self.param['srcName'], 'Source name')
        hdu.header['NIGHT'] = (self.param['nightName'], 'Night Name')
        hdu.header['SRCGEOM'] = (self.param['srcGeometry'], 'Source Aperture Geometry')
        hdu.header['BKGGEOM'] = (self.param['bkgGeometry'], 'Background Aperture Geometry')
        if 'apRadius' in self.param:
            hdu.header['APRADIUS'] = (self.param['apRadius'], 'Aperture radius (px)')
        elif 'apHeight' in self.param:
            hdu.header['APRADIUS'] = (self.param['apHeight'], 'Aperture radius (px)')
            hdu.header['APHEIGHT'] = (self.param['apHeight'], 'Aperture radius (px)')
            hdu.header['APWIDTH'] = (self.param['apWidth'], 'Aperture radius (px)')
        else:
            print("No apHeight or apRadius found in parameters")
        hdu.header['SCALEDAP'] = (self.param['scaleAperture'], 'Is the aperture scaled by the FWHM?')
        hdu.header['APSCALE'] = (self.param['apScale'], 'If scaling apertures, which scale factor?')
        
#        hdu.header[''] = '       Background Subtraction parameters '
        hdu.header['BKGSUB'] = (self.param['bkgSub'], 'Do a background subtraction?')
        hdu.header['BKGSTART'] = (self.param['backStart'], 'Background Annulus start (px), if used')
        hdu.header['BKGEND'] = (self.param['backEnd'], 'Background Annulus end (px), if used')
        if 'backHeight' in self.param:
            hdu.header['BKHEIGHT'] = (self.param['backHeight'], 'Background Box Height (px)')
            hdu.header['BKWIDTH'] = (self.param['backWidth'], 'Background Box Width (px)')
        hdu.header['BKOFFSTX'] = (self.param['backOffset'][0], 'X Offset between background and source (px)')
        hdu.header['BKOFFSTY'] = (self.param['backOffset'][1], 'Y Offset between background and source (px)')
        hdu.header['BKGMETH'] = (self.param['bkgMethod'], 'Background subtraction method')
        if self.param['bkgMethod'] == 'colrow':
            hdu.header['BKGDIREC'] = (" ".join(self.param['backsub_directions']), 'The directions, in order, for polynomial background sub')
            hdu.header['BKGORDRX'] = (self.param['bkgOrderX'], 'X Background subtraction polynomial order')
            hdu.header['BKGORDRY'] = (self.param['bkgOrderY'], 'Y Background subtraction polynomial order')
        
#        hdu.header[''] = '       Centroiding Parameters '
        hdu.header['BOXSZ'] = (self.param['boxFindSize'], 'half-width of the box used for centroiding')
        hdu.header['COPYCENT'] = (self.param['copyCentroidFile'], 'Name of the file where centroids are copied (if used)')
                
#        hdu.header[''] = '       Timing Parameters '
        hdu.header['JDREF'] = (self.param['jdRef'], ' JD reference offset to subtract for plots')
        
#        hdu.header[''] = '       Image Parameters '
        hdu.header['ISCUBE'] = (self.param['isCube'], 'Is the image data 3D?')
        hdu.header['CUBPLANE'] = (self.param['cubePlane'], 'Which plane of the cube is used?')
        hdu.header['DOCEN'] = (self.param['doCentering'], 'Is each aperture centered individually?')
        hdu.header['EXTNAMEU'] = (self.param['FITSextension'], 'FITS extension used of data')
        hdu.header['NANTREAT'] = (self.param['nanTreatment'], 'How are NaN pixels treated?')
        hdu.header['SLOPEIMG'] = (self.param['isSlope'], 'Are original images slopes, then multiplied by int time?')
        hdu.header['SUBPIXEL'] = (self.param['subpixelMethod'], 'Treatment of apertures at the subpixel level')
        

        
        hduFileNames = self.make_filename_hdu()
        
        hduTime = fits.ImageHDU(jdArr)
        hduTime.header['UNITS'] = ('days','JD time, UT')
        
        hduErr = fits.ImageHDU(data=errArr,header=basicHeader)
        hduErr.name = 'Phot Err'
        
        hduBack = fits.ImageHDU(data=backArr,header=basicHeader)
        hduBack.name = 'Backg Phot'
        
        hduCen = fits.ImageHDU(data=self.cenArr,header=self.cenHead)
        
        hdu.name = 'Photometry'
        hduTime.name = 'Time'
        hduCen.name = 'Centroids'
        ## hduFileName.name = 'Filenames' # already named by make_filename_hdu
        
        ## Get an example original header
        exImg, exHeader = self.get_default_im()
        hduOrigHeader = fits.ImageHDU(None,exHeader)
        hduOrigHeader.name = 'Orig Header'
        
        HDUList = fits.HDUList([hdu,hduErr,hduBack,hduTime,hduCen,hduFileNames,hduOrigHeader])
        
        if self.keepFWHM == True:
            hduFWHM = fits.ImageHDU(self.fwhmArr,header=self.headFWHM)
            HDUList.append(hduFWHM)
        
        HDUList.writeto(self.photFile,overwrite=True)
        warnings.resetwarnings()
    
    def plot_phot(self,offset=0.,refCorrect=False,ax=None,fig=None,showLegend=True,
                  normReg=None,doBin=None,doNorm=True,yLim=[None,None],
                  excludeSrc=None,errBar=None,showPlot=False):
        """ Plots previously calculated photometry 
        
        Parameters
        ---------------------
        offset : float
            y displacement for overlaying time series
        refCorrect : bool
            Use reference star-corrected photometry?
        ax : matplotlib axis object
            If the axis was created separately, use the input axis object
        fig : matplotlib figure object
            If the figure was created separately, use the input axis object
        showLegend : bool
            Show a legend?
        normReg: list with two items or None
            Relative region over which to fit a baseline and re-normalize. This only works on reference-corrected photometry for now
        doBin : float or None
            The bin size if showing binned data. This only works on reference-corrected photometry for now
        doNorm : bool
            Normalize the individual time series?
        yLim:  List
            List of Y limit to show
        errBar : string or None
            Describes how error bars will be displayed. None=none, 'all'=every point,'one'=representative
        excludeSrc : List or None
            Custom sources to exclude in the averaging (to exclude specific sources in the reference time series).
            For example, for 5 sources, excludeSrc = [2] will use [1,3,4] for the reference
        showPlot: bool
            Show the plot? Otherwise, it saves it to a file
        """
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        errArr = HDUList['PHOT ERR'].data
        head = photHDU.header
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        timeHead = jdHDU.header
        
        jdRef = self.param['jdRef']
        
        if ax == None:
            fig, ax = plt.subplots()
        
        if refCorrect == True:
            yCorrected, yCorrected_err = self.refSeries(photArr,errArr,excludeSrc=excludeSrc)
            x = jdArr - jdRef
            if normReg == None:
                yShow = yCorrected
            else:
                fitp = (x < normReg[0]) | (x > normReg[1])
                polyBase = robust_poly(x[fitp],yCorrected[fitp],2,sigreject=2)
                yBase = np.polyval(polyBase,x)
                
                yShow = yCorrected / yBase
                
            if errBar == 'all':
                ax.errorbar(x,yShow,label='data',marker='o',linestyle='',markersize=3.,yerr=yCorrected_err)
            else:
                ax.plot(x,yShow,label='data',marker='o',linestyle='',markersize=3.)
            
                madY = np.nanmedian(np.abs(yShow - np.nanmedian(yShow)))
                if errBar == 'one':
                    ax.errorbar([np.median(x)],[np.median(yShow) - 4. * madY],
                                yerr=np.median(yCorrected_err),fmt='o',mfc='none')
            #pdb.set_trace()
            
            if doBin is not None:
                minValue, maxValue = 0.95, 1.05 ## clip for cosmic rays
                goodP = (yShow > minValue) & (yShow < maxValue)
                nBin = int(np.round((np.max(x[goodP]) - np.min(x[goodP]))/doBin))
                
                if nBin > 1:
                    yBins = Table()
                    for oneStatistic in ['mean','std','count']:
                        if oneStatistic == 'std':
                            statUse = np.std
                        else: statUse = oneStatistic
                        
                        yBin, xEdges, binNum = binned_statistic(x[goodP],yShow[goodP],
                                                                statistic=statUse,bins=nBin)
                        yBins[oneStatistic] = yBin
                
                    ## Standard error in the mean
                    stdErrM = yBins['std'] / np.sqrt(yBins['count'])
                
                    xbin = (xEdges[:-1] + xEdges[1:])/2.
                    ax.errorbar(xbin,yBins['mean'],yerr=stdErrM,marker='s',markersize=3.,
                            label='binned')
                
        else:
            for oneSrc in range(self.nsrc):
                yFlux = photArr[:,oneSrc]
                yNorm = yFlux / np.nanmedian(yFlux)
                if oneSrc == 0:
                    pLabel = 'Src'
                else:
                    pLabel = 'Ref '+str(oneSrc)
                if doNorm == True:
                    yplot = yNorm - offset * oneSrc
                else:
                    yplot = yFlux - offset * oneSrc
                
                ## To avoid repeat colors, switch to dashed lins
                if oneSrc >= 10: linestyle='dashed'
                else: linestyle= 'solid'
                ax.plot(jdArr - jdRef,yplot,label=pLabel,linestyle=linestyle)
        
            if head['SRCGEOM'] == 'Circular':
                ax.set_title('Src Ap='+str(head['APRADIUS'])+',Back=['+str(head['BKGSTART'])+','+
                             str(head['BKGEND'])+']')
        ax.set_xlabel('JD - '+str(jdRef))
        ax.set_ylim(yLim[0],yLim[1])
        if doNorm == True:
            ax.set_ylabel('Normalized Flux + Offset')
        else:
            ax.set_ylabel('Flux + Offset')
        #ax.set_ylim(0.94,1.06)
        if showLegend == True:
            ax.legend(loc='best',fontsize=10)
        
        if showPlot == True:
            fig.show()
        else:
            if refCorrect == True:
                outName = 'tser_refcor/refcor_{}.pdf'.format(self.dataFileDescrip)
                outPath = os.path.join(self.baseDir,'plots','photometry',outName)
            else:
                outName = 'raw_tser_{}.pdf'.format(self.dataFileDescrip)
                outPath = os.path.join(self.baseDir,'plots','photometry','tser_allstar',outName)
            fig.savefig(outPath)
            plt.close(fig)
        
        HDUList.close()
        
        
    
    def print_phot_statistics(self,refCorrect=True,excludeSrc=None,shorten=False,
                              returnOnly=False,removeLinear=True):
        """
        Print the calculated and theoretical noise as a table
                              
        Parameters
        ----------
        refCorrect: bool
            Use reference stars to correct target?
            If True, there is only one row in the table for the target.
            If False, there is a row for each star's absolute noise
        excludeSrc: list, or None
            A list of sources (or None) to exclude as reference stars
            Given by index number
        shorten: bool
            Shorten the number of points used the time series?
            Useful if analyzing the baseline befor transit, for example.
        returnOnly: bool
            If True, a table is returned.
            If False, a table is printed and another is returned
        
        removeLinear: bool
            Remove a linear trend from the data first?
        """
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        head = photHDU.header
        timeArr = HDUList['TIME'].data
        errArr = HDUList['PHOT ERR'].data
        
        t = Table()
        if (head['NSOURCE'] == 1) & (refCorrect == True):
            warnings.warn('Only once source, so defaulting to refCorrect=False')
            refCorrect = False
        
        if shorten == True:
            photArr = photArr[0:15,:]
            nImg = 15
        else:
            nImg = self.nImg
        
        if refCorrect == True:
            yCorrected, yCorrected_err = self.refSeries(photArr,errArr,
                                                        excludeSrc=excludeSrc)
            
            if removeLinear == True:
                xNorm = (timeArr - np.min(timeArr))/(np.max(timeArr) - np.min(timeArr))
                poly_fit = robust_poly(xNorm,yCorrected,1)
                yCorrected = yCorrected / np.polyval(poly_fit,xNorm)
            
            if shorten == True:
                yCorrected = yCorrected[0:15]
            
            t['Stdev (%)'] = np.round([np.nanstd(yCorrected) * 100.],4)
            t['Theo Err (%)'] = np.round(np.nanmedian(yCorrected_err) * 100.,4)
            mad = np.nanmedian(np.abs(yCorrected - np.nanmedian(yCorrected)))
            t['MAD (%)'] = np.round(mad * 100.,4)
        else:
            t['Source #'] = np.arange(self.nsrc)
            medFlux = np.nanmedian(photArr,axis=0)
            t['Stdev (%)'] = np.round(np.nanstd(photArr,axis=0) / medFlux * 100.,4)
            t['Theo Err (%)'] = np.round(np.nanmedian(errArr,axis=0) / medFlux * 100.,4)
            tiledFlux = np.tile(medFlux,[nImg,1])
            mad = np.nanmedian(np.abs(photArr - tiledFlux),axis=0) / medFlux
            t['MAD (%)'] = np.round(mad * 100.,4)
        
        if returnOnly:
            pass
        else:
            print(t)
        
        HDUList.close()
        return t
    
    def plot_state_params(self,excludeSrc=None):
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        head = photHDU.header
        errArr = HDUList['PHOT ERR'].data
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        t = jdArr - np.round(np.min(jdArr))
        timeHead = jdHDU.header
        
        cenData = HDUList['CENTROIDS'].data
        fwhmData = HDUList['FWHM'].data
        
        backData = HDUList['BACKG PHOT'].data
        
        fig, axArr = plt.subplots(7,sharex=True)
        yCorr, yCorr_err = self.refSeries(photArr,errArr,excludeSrc=excludeSrc)
        axArr[0].plot(t,yCorr)
        axArr[0].set_ylabel('Ref Cor F')
        
        
        for oneSrc in range(self.nsrc):
            yFlux = photArr[:,oneSrc]
            axArr[1].plot(t,yFlux / np.median(yFlux))
            axArr[1].set_ylabel('Flux')
            xCen = cenData[:,oneSrc,0]
            backFlux = backData[:,oneSrc]
            axArr[2].plot(t,backFlux / np.median(backFlux))
            axArr[2].set_ylabel('Back')
            axArr[3].plot(t,xCen - np.median(xCen))
            axArr[3].set_ylabel('X Pos')
            yCen = cenData[:,oneSrc,1]
            axArr[4].plot(t,yCen - np.median(yCen))
            axArr[4].set_ylabel('Y Pos')
            fwhm1 = fwhmData[:,oneSrc,0]
            axArr[5].plot(t,np.abs(fwhm1))
            axArr[5].set_ylabel('FWHM 1')
            fwhm2 = fwhmData[:,oneSrc,1]
            axArr[6].plot(t,np.abs(fwhm1))
            axArr[6].set_ylabel('FWHM 2')
        
        fig.show()
    
    def plot_flux_vs_pos(self,refCorrect=True):
        """
        Plot flux versus centroid to look for flat fielding effects
        """
        HDUList = fits.open(self.photFile)
        if refCorrect == True:
            yNorm, yErrNorm = self.refSeries(HDUList['PHOTOMETRY'].data,HDUList['PHOT ERR'].data)
        else:
            yFlux = HDUList['PHOTOMETRY'].data[:,0]
            yNorm = yFlux / np.median(yFlux)
        
        cenX = HDUList['CENTROIDS'].data[:,0,0]
        cenY = HDUList['CENTROIDS'].data[:,0,1]
        
        fig, axArr = plt.subplots(1,2,sharey=True,figsize=(9,4.5))
        
        for ind,oneDir, coord in zip([0,1],['X','Y'],[cenX,cenY]):
            axArr[ind].plot(coord,yNorm,'o')
            axArr[ind].set_xlabel('{} (px)'.format(oneDir))
            axArr[ind].set_ylabel('Norm F')
        
        #yPoly = 
        
        fig.show()
        HDUList.close()
    
    def refSeries(self,photArr,errPhot,reNorm=False,excludeSrc=None,sigRej=5.):
        """ Average together the reference stars
        
        Parameters
        -------------
        reNorm: bool
            Re-normalize all stars before averaging? If set all stars have equal weight. Otherwise, the stars are summed together, which weights by flux
        excludeSrc: arr
            Custom sources to use in the averaging (to exclude specific sources in the reference time series. For example, for 5 sources, excludeSrc = [2] will use [1,3,4] for the reference
        sigRej: int
            Sigma rejection threshold
        """
        combRef = []
        
        srcArray = np.arange(self.nsrc,dtype=np.int)
        
        if excludeSrc == None:
            maskOut = (srcArray == 0)
        else:
            maskOut = np.zeros(self.nsrc,dtype=bool)
            maskOut[0] = True
            for oneSrc in excludeSrc:
                if (oneSrc < 0) | (oneSrc >= self.nsrc):
                    pdb.set_trace()
                    raise Exception("{} is an invalid source among {}".format(oneSrc,self.nsrc))
                else:
                    maskOut[oneSrc] = True
        
        refMask2D = np.tile(maskOut,(self.nImg,1))
        ## also mask points that are NaN
        nanPt = (np.isfinite(photArr) == False)
        refMask2D = refMask2D | nanPt
        refPhot = np.ma.array(photArr,mask=refMask2D)
        
        ## Normalize all time series
        norm1D_divisor = np.nanmedian(photArr,axis=0)
        norm2D_divisor = np.tile(norm1D_divisor,(self.nImg,1))
        
        normPhot = refPhot / norm2D_divisor
        normErr = errPhot / norm2D_divisor
        
        ## Find outliers
        # Median time series
        medTimSeries1D = np.nanmedian(normPhot,axis=1)
        medTimSeries2D = np.tile(medTimSeries1D,(self.nsrc,1)).transpose()
        
        # Absolute deviation
        absDeviation = np.abs(normPhot - medTimSeries2D)
        # Median abs deviation of all reference photometry
        MADphot = np.nanmedian(absDeviation)
        # Points that deviate above threshold
        badP = (absDeviation > sigRej * np.ones((self.nImg,self.nsrc),dtype=np.float) * MADphot)
        
        normPhot.mask = refMask2D | badP
        refPhot.mask = refMask2D | badP
        
        if reNorm == True:
            ## Weight all stars equally
            combRef = np.nanmean(normPhot,axis=1)
            combErr = np.sqrt(np.nansum(normErr**2,axis=1)) / (self.nsrc - np.sum(maskOut))
            
        else:
            ## Weight by the flux, but only for the valid points left
            weights = np.ma.array(norm2D_divisor,mask=normPhot.mask)
            ## Make sure weights sum to 1.0 for each time point (since some sources are missing)
            weightSums1D = np.nansum(weights,axis=1)
            weightSums2D = np.tile(weightSums1D,(self.nsrc,1)).transpose()
            weights = weights / weightSums2D
            combRef = np.nansum(normPhot * weights,axis=1)
            combErr = np.sqrt(np.nansum((normErr * weights)**2,axis=1)) / np.nansum(weights,axis=1)
            
        
        yCorrected = photArr[:,0] / combRef
        yCorrNorm = yCorrected / np.nanmedian(yCorrected)
        
        yErrNorm = np.sqrt(normErr[:,0]**2 + (combErr/ combRef)**2)
        
        return yCorrNorm, yErrNorm
    
    def get_tSeries(self):
        """
        Get a simple table of the photometry after extraction
        
        Returns
        --------
        t1: astropy.table object
            A table of fluxes and time
        t2: astropy.table object
            A table of flux errors and time
        """
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        errArr = HDUList['PHOT ERR'].data
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        
        t1, t2 = Table(), Table()
        t1['Time (JD)'] = jdArr
        t2['Time (JD)'] = jdArr
        for oneSrc in np.arange(self.nsrc):
            t1['Flux {}'.format(oneSrc)] = photArr[:,oneSrc]
            t2['Error {}'.format(oneSrc)] = errArr[:,oneSrc]
        HDUList.close()
        
        return t1, t2
    
    def get_refSeries(self,excludeSrc=None):
        """
        Get the reference-corrected time series
        
        Parameters
        -----------
        excludeSrc: list or None
            Numbers of the reference stars to exclude
        
        Returns
        --------
        t: numpy array
            time (JD - reference)
        yCorr: numpy array
            Reference-corrected time series
        yCorr_err: numpy array
            Reference-corrected time series error
        """
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        errArr = HDUList['PHOT ERR'].data
        
        yCorr, yCorr_err = self.refSeries(photArr,errArr,excludeSrc=excludeSrc)
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        t = jdArr - np.round(np.min(jdArr))
        
        HDUList.close()
        
        return t, yCorr, yCorr_err
        
    def interactive_refSeries(self,excludeSrc=None,
                              refCorrect=True,srcInd=0):
        """
        Plot a bokeh interactive plot of the photometry
        This lets you see which images are outliers
        
        Parameters
        ----------
        refCorrect: bool
            Reference correct the time series?
        excludeSrc: list or None
            Which sources to exclude from reference series
        srcInd: int
            Which source index to plot if refCorrect is False
        """
        if refCorrect == True:
            t, yCorr, yCorr_err = self.get_refSeries(excludeSrc=excludeSrc)
            outName = "refseries_{}.html".format(self.dataFileDescrip)
            
        else:
            t1, t2 = self.get_tSeries()
            t = t1['Time (JD)']
            yCorr = t1['Flux {}'.format(srcInd)]
            yCorr_err = t2['Error {}'.format(srcInd)]
            outName = "abseries_{}.html".format(self.dataFileDescrip)
        
        outFile = os.path.join(self.baseDir,'plots','photometry','interactive',outName)
        
        fileBaseNames = []
        fileTable = Table.read(self.photFile,hdu='FILENAMES')
        indexArr = np.arange(len(fileTable))
        for oneFile in fileTable['File Path']:
            fileBaseNames.append(os.path.basename(oneFile))
        
        bokeh.plotting.output_file(outFile)
        dataDict = {'t': t,'y':yCorr,'name':fileBaseNames,'ind':indexArr}
        source = ColumnDataSource(data=dataDict)
        p = bokeh.plotting.figure()
        p.background_fill_color="#f5f5f5"
        p.grid.grid_line_color="white"
        p.circle(x='t',y='y',source=source)
        p.add_tools(HoverTool(tooltips=[('name', '@name'),('index','@ind')]))
        bokeh.plotting.show(p)
        

    def getImg(self,path):
        """ Load an image from a given path and extensions"""
        ext = self.param['FITSextension']
        headExtension = self.param['HEADextension']
        HDUList = fits.open(path)
        data = HDUList[ext].data
        if self.param['isCube'] == True:
            img = data[self.param['cubePlane'],:,:]
        else:
            img = data
        
        if self.param['nanTreatment'] == 'zero':
            nanPt = (np.isfinite(img) == False)
            img[nanPt] = 0.0
        elif self.param['nanTreatment'] == 'leave':
            pass
        elif self.param['nanTreatment'] == 'value':
            nanPt = (np.isfinite(img) == False)
            img[nanPt] = self.param['nanReplaceValue']
        else:
            raise NotImplementedError
        
        head = HDUList[headExtension].header
        if self.param['isSlope'] == True:
            itimeKey = self.param['itimeKeyword']
            if itimeKey in head:
                intTime = head[itimeKey]
            elif 'EFFINTTM' in head:
                intTime = head['EFFINTTM']
            else:
                warnings.warn("Couldn't find {} in header. Trying EXPTIME".format(itimeKey))
                intTime = head['EXPTIME']
            ## If it's a slope image, multiply rate time intTime to get counts
            img = img * intTime
        
        if self.param['detectorGain'] != None:
            img = img * self.param['detectorGain']
        
        HDUList.close()
        return img, head
        
    def save_phot(self):
        """ Save a reference-corrected time series 
        """
        t = Table()
        
                
        HDUList = fits.open(self.photFile)
        ## Grab the metadata from the header
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        errArr = HDUList['PHOT ERR'].data
        head = photHDU.header
        head.pop('NAXIS')
        head.pop('NAXIS1')
        head.pop('NAXIS2')
        head.pop('SIMPLE')
        head.pop('BITPIX')
        
        t.meta = head
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        
        t['Time'] = jdArr
        t['Y Corrected'], yCorrErr = self.refSeries(photArr,errArr)
        
        if 'PHOT ERR' in HDUList:
            ## Error from the photometry point
            hduErr = HDUList['PHOT ERR']
            t['Y Corr Err'] = hduErr.data[:,0] / photArr[:,0]
        else:
            print("Warning, No Error recorded for {}".format(self.photFile))
        ## FOr now this ignores errors in the reference stars
        ## To Do: add in error from reference stars
        
        t.write(self.refCorPhotFile,overwrite=True)

class batchPhot:
    """ 
    Create several photometry objects and run phot over all of them
    """
    def __init__(self,batchFile='parameters/phot_params/example_batch_phot.yaml'):
        self.alreadyLists = {'refStarPos': 2,'backOffset': 1,'apRange': 1,'excludeList': 1,
                             'backsub_directions': 1}
        self.general_init(batchFile=batchFile)
    
    def general_init(self,batchFile='parameters/phot_params/example_batch_phot.yaml'):
        self.batchFile = batchFile
        self.batchParam = read_yaml(batchFile)
        
        ## Find keys that are lists. These are ones that are being run in batches
        ## However, a few keywords are already lists (like [x,y] coordinates))
        # and we are looking to see if those are lists of lists
        ## the self.alreadyLists dictionary specifies the depth of the list
        ## it could be a list of a list of list
        self.paramLists = []
        self.counts = []
        for oneKey in self.batchParam.keys():
            if oneKey in self.alreadyLists:
                depth = self.alreadyLists[oneKey]
                value = deepcopy(self.batchParam[oneKey])
                ## Make sure the parameter is not None, so we won't be digging
                if value != None:
                    ## dig as far as the depth number to check for a list
                    for oneDepth in np.arange(depth):
                        value = value[0]
                if type(value) == list:
                    self.paramLists.append(oneKey)
                    self.counts.append(len(self.batchParam[oneKey]))
            else:
                if type(self.batchParam[oneKey]) == list:
                    self.paramLists.append(oneKey)
                    self.counts.append(len(self.batchParam[oneKey]))
        if len(self.counts) == 0:
            raise Exception("No lists found to iterate over")
        self.NDictionaries = self.counts[0]
        ## Make sure there are a consistent number of parameters in each list
        if np.all(np.array(self.counts) == self.NDictionaries) == False:
            descrip1 = "Inconsistent parameter counts in {}.".format(self.batchFile)
            descrip2 = "Parameters {} have counts {}".format(self.paramLists,self.counts)
            raise Exception(descrip1+descrip2)
        
        self.paramDicts = []
        for ind in np.arange(self.NDictionaries):
            thisDict = {}
            for oneKey in self.batchParam.keys():
                if oneKey in self.paramLists:
                    thisDict[oneKey] = self.batchParam[oneKey][ind]
                else:
                    thisDict[oneKey] = self.batchParam[oneKey]
            self.paramDicts.append(thisDict)
            
    def print_all_dicts(self):
        print(yaml.dump(self.paramDicts,default_flow_style=False))
    
    def make_pipe_obj(self,directParam):
        """
        Make a photometry pipeline object that will be executed in batch
        """
        return phot(directParam=directParam)
    
    def batch_run(self,method,**kwargs):
        """
        Run any method of the photometry class by name. This will 
        cycle through all parameter lists and run the method
        
        Parameters
        -----------
        method: str
           Photometry method to run in batch mode
        
        **kwargs: keywords or dict
           Arguments to be passed to this method
        
        Returns
        -------
        batch_result: list or None
            A list of results from the photometry method.
            If all results are None, then batch_run returns None
        
        """
        batch_result = []
        for oneDict in self.paramDicts:
            thisPhot = self.make_pipe_obj(oneDict)
            print("Working on {} for batch {} {} ".format(method,
                                                         thisPhot.param['srcName'],
                                                         thisPhot.dataFileDescrip))
            photMethod = getattr(thisPhot,method)
            result = photMethod(**kwargs)
            batch_result.append(result)
        if all(v is None for v in batch_result):
            batch_result = None
        
        return batch_result
    
    def run_all(self,useMultiprocessing=False):
        self.batch_run('showStarChoices',showAps=True,srcLabel='0')
        self.batch_run('do_phot',useMultiprocessing=useMultiprocessing)

    def plot_all(self):
        self.batch_run('plot_phot')
    
    def test_apertures(self):
        self.batch_run('showStarChoices',showAps=True,srcLabel='0')

    def return_phot_obj(self,ind=0):
        """
        Return a photometry object so other methods and attributes can be explored
        """
        return phot(directParam=self.paramDicts[ind])


def ensure_directories_are_in_place(filePath):
    """
    Takes a name of a file and makes sure the directories are there to save the file
    """
    dirPath = os.path.split(filePath)[0]
    
    if dirPath != "":
        if os.path.exists(dirPath) == False:
            print("Creating {} for file output".format(dirPath))
            os.makedirs(dirPath)

def get_tshirt_example_data():
    """
    Download all example tshirt data. This is needed to run tests and
    the default parameter files
    """
    baseDir = get_baseDir()
    data_list_file = resource_filename('tshirt','directory_info/example_data_list.txt')
    with open(data_list_file) as dlf:
        fileList = dlf.read().splitlines()
    onlineDir = 'https://raw.githubusercontent.com/eas342/tshirt_example_data/main/'
    
    example_tshirt_dir = os.path.join(baseDir,'example_tshirt_data','example_data')
    for oneFile in fileList:
        onlinePath = onlineDir + str(oneFile) + '?raw=true'
        #print('Online Path: {}'.format(onlinePath))
        
        outPath = os.path.join(example_tshirt_dir,oneFile)
        ensure_directories_are_in_place(outPath)
        
        if os.path.exists(outPath) == False:
            logging.info("Attempting to download {}".format(onlinePath))
            
            if sys.version > '3':
                f = urllib.request.urlopen(onlinePath)
            else:
                f = urllib.urlopen(onlinePath)
            with open(outPath,'wb') as f_out:
                f_out.write(f.read())
            

class prevPhot(phot):
    """
    Loads in previous photometry from FITS data. Inherits functions from the phot class
    
    Parameters
    ---------------
    photFile: str
        Directory of the photometry file
    """
    def __init__(self,photFile='tser_data/phot/phot_kic1255_UT2016_06_12.fits'):
        self.photFile = photFile
        HDUList = fits.open(self.photFile)
        photHead = HDUList[0].header
        
        self.fileL = 'origProcFiles'
        
        self.nImg = photHead['NIMG']
        
        #xCoors, yCoors = [], []
        #positions = self.param['refStarPos']
        self.nsrc = photHead['NSOURCE']
        
        #self.srcApertures = CircularAperture(positions,r=self.param['apRadius'])
        #self.xCoors = self.srcApertures.positions[:,0]
        #self.yCoors = self.srcApertures.positions[:,1]
        #self.bkgApertures = CircularAnnulus(positions,r_in=self.param['backStart'],r_out=self.param['backEnd'])
        self.srcNames = np.array(np.arange(self.nsrc),dtype=str)
        self.srcNames[0] = 'src'
        
        self.dataFileDescrip = os.path.splitext(os.path.basename(self.photFile))[0]
                
        self.photHead = photHead
        self.param = {}
        
        keywordPath = os.path.join(os.path.dirname(__file__), '..', 'parameters','phot_params',
                                   'keywords_for_phot_pipeline.csv')
        photKeywords = ascii.read(keywordPath)
        for ind,oneKeyword in enumerate(photKeywords['FITS Keyword']):
            if oneKeyword in self.photHead:
                self.param[photKeywords[ind]['Parameter Name']] = self.photHead[oneKeyword]
            else:
                self.param[photKeywords[ind]['Parameter Name']] = photKeywords[ind]['Default Value']
        
        ## Some parameters are more complicated
        if 'BKOFFSTX' in self.photHead and 'BKOFFSTY' in self.photHead:
            self.param['backOffset'] = [self.photHead['BKOFFSTX'],self.photHead['BKOFFSTY']]
        
        self.centroidFile = self.photFile
        
        positions = HDUList['CENTROIDS'].data[self.nImg // 2]
        
        self.set_up_apertures(positions)
        
        self.yCoors = self.srcApertures.positions[:,1]
        
        HDUList.close()
        
        self.refCorPhotFile = 'tser_data/refcor_phot/refcor_'+self.dataFileDescrip+'.fits'

def saveRefSeries():
    """ Saves the reference-corrected time series for all nights"""
    fileL = glob.glob('tser_data/phot/phot_kic1255_UT????_??_??.fits')
    for oneFile in fileL:
        phot = prevPhot(photFile=oneFile)
        phot.save_phot()
    

def allTser(refCorrect=False,showBestFit=False):
    """ Plot all time series for KIC 1255 
    
    Parameters
    -----------
    refCorrect: bool
        Do the reference correction?
    showBestFit: bool
        Show the best fit? If true, shows the best fit.
        If false, it shows the average Kepler light curve.
        This only works after the best fits have been previously
        calculated.
    """
    
    allFits = glob.glob('tser_data/phot/phot_kic1255_UT????_??_??.fits')
    epochs = [2457551.8250368, 2457553.785697 ,  2457581.8884932,
              2457583.8491534,  2457585.8098136]
    periodP = 0.6535534
    nNights = len(allFits)
    
    fig = plt.figure(figsize=(8,6))
    nY, nX = (3,2)
    gs = gridspec.GridSpec(nY,nX,wspace=0.0,hspace=0.7)
    yArr, xArr = np.mgrid[0:nY,0:nX]
    yRavel = yArr.ravel()
    xRavel = xArr.ravel()
    
    kepHDUList = fits.open('tser_data/reference_dat/avg_bin_kep.fits')
    kepLC = kepHDUList[1].data
    tKepler = kepLC['BINMID'] * kepHDUList[1].header['PERIOD'] ## time in days
    fKepler = kepLC['YBIN']
    
    bestFitDir = 'tser_data/best_fit_models/kic1255/'
    
    for ind, oneFits in enumerate(allFits):
        phot = prevPhot(photFile=oneFits)
        thisAx = plt.subplot(gs[ind])
        normCen = epochs[ind] - phot.param['jdRef']
        if showBestFit == True:
            ## If showing the best fit, do not detredn to illustrtate the 
            ## baseline fit & everything
            normReg = None
        else:
            ## Remove the baseline
            normReg = [normCen - 0.04,normCen+0.04]
        
        phot.plot_phot(ax=thisAx,fig=fig,showLegend=False,refCorrect=refCorrect,
                       normReg=normReg,doBin=20./(60. * 24))
        if refCorrect == True:
            if showBestFit == True:
                ## leave enough room to show baseline fit
                thisAx.set_ylim(0.99,1.015)
            else:
                thisAx.set_ylim(0.99,1.01)
        else: thisAx.set_ylim(0.9,1.1)
        thisAx.set_xlim(0.65,0.99)
        
        fitsStyleTitle = phot.param['nightName'].replace('_','-')
        aasStyleTitle = phot.param['nightName'].replace('_',' ')
        aasStyleTitle = aasStyleTitle.replace('07','Jul')
        aasStyleTitle = aasStyleTitle.replace('06','Jun')
        thisAx.set_title(aasStyleTitle)
        ## Hide the y labels for stars not on the left. Also shorten title
        if xRavel[ind] == 0:
            thisAx.set_ylabel('Norm Fl')
        else:
            thisAx.yaxis.set_visible(False)
        
        ## Show transits for reference corrected photometry
        if refCorrect == True:
            thisAx.axvline(x=epochs[ind] - phot.param['jdRef'],linewidth=2,color='red',alpha=0.5)
            if showBestFit == True:
                dat = ascii.read('{}kic1255_best_fit_{}.csv'.format(bestFitDir,phot.param['nightName']))
                xReference = dat['Phase'] * periodP + epochs[ind] - phot.param['jdRef']
                yReference = dat['Best Fit Y']
                referenceName = 'Best Fit'
            else:
                ## Over-plot Kepler Avg SC Light curves
                xReference = tKepler + epochs[ind] - phot.param['jdRef']
                yReference = fKepler
                referenceName = 'Kepler SC Avg'
            
            thisAx.plot(xReference,yReference,color='green',linewidth=2,
                        alpha=0.5,label=referenceName)
        
        ## Show a legend with the last one
        if ind == len(allFits) - 1:
            thisAx.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=2)
        
    
    fig.show()
    if refCorrect == True:
        outPath = os.path.join(self.baseDir,'plots','photometry','tser_refcor','all_kic1255.pdf')
    else:
        outPath = os.path.join(self.baseDir,'plots','photometry','tser_allstar','all_kic1255.pdf')
    fig.savefig(outPath)

def ensure_coordinates_are_within_bounds(xCoord,yCoord,img):
    """
    Check if x and y coordinates are inside an image
    If they are are, spit them back.
    Otherwise, give the coordinates at the closest x/y edge
    
    Parameters
    ----------
    """
    assert len(img.shape) == 2, "Should have a 2D image"
    
    xmin, ymin = 0, 0
    xmax = img.shape[1] - 1
    ymax = img.shape[0] - 1
    out_x = np.minimum(np.maximum(xCoord,0),xmax)
    out_y = np.minimum(np.maximum(yCoord,0),ymax)

    return out_x,out_y

def do_binning(x,y,nBin=20):
    """
    A function that uses scipy binned_statistic to bin data
    
    It also calculates the standard error in each bin,
    which can be used as an error estimate
    
    Parameters
    --------------
    x: numpy array
        Independent variable for use in assigning data to bins
    y: numpy array
        Dependent variable to be binned
    
    Returns
    -------------
    3 item tuple:
    xBin, yBin, yStd
    
    xBin: numpy array
        Middles of the bins
    yBin: numpy array
        mean value in bin
    yStd: numpy array
        standard error of each bin
    """
    yBins = Table()
    for oneStatistic in ['mean','std','count']:
        yBin, xEdges, binNum = binned_statistic(x,y,
                                                statistic=oneStatistic,bins=nBin)
        yBins[oneStatistic] = yBin

    ## Standard error in the mean
    stdErrM = yBins['std'] / np.sqrt(yBins['count'])
    xShow = (xEdges[:-1] + xEdges[1:])/2.
    yShow = yBins['mean']
    yErrShow = stdErrM

    return xShow, yShow, yErrShow

def allan_variance(x,y,yerr=None,removeLinear=False,yLim=[None,None],
                   binMin=50,binMax=2000,customShortName=None,
                   logPlot=True,clip=False,xUnit='min',
                   yUnit='ppm',showPlot=False):
    """
    Make an Allan Variance plot for a time series
    to see if it bins as sqrt(N) statistics
    
    Parameters
    ------------
    x: numpy array
        Independent variable
    y: numpy array
        Dependent variable like flux
    (optional keywords)
    yerr: numpy array
        Theoretical error
    customShortName: str
        Name for file
    yLim: 2 element list
        Specify custom Y values for the plot
    binMin: int
        Bin size for the smallest # of bins
    binMax: int
        Bin size for the largest # of bins
    removeLinear: bool
        Remove a linear trend from the time series first?
    clip: bool
        Clip the first few points?
    xUnit: str
        Name of units for X axis of input series
    yUnit: str
        Name of units for Y axis to be binned
    showPlot: bool
        Render the plot with matplotlib? If False, it is saved instead.
    """

    if clip == True:
        x = x[2:]
        y = y[2:]
        if yerr is not None:
            y = y[2:]
        print("clipping to {} points".format(len(x)))

    if removeLinear == True:
        yPoly = robust_poly(x,y,1)
        ymod = np.polyval(yPoly,x)
        y = y / ymod
    
    if yUnit == 'ppm':
        y = y * 1e6
        yerr = yerr * 1e6
    nPt = len(y)
    
    logBinNums = np.linspace(np.log10(binMin),np.log10(binMax),20)
    binNums = np.array(10**logBinNums,dtype=np.int)
    
    binSizes, stds, theoNoise, wNoise = [], [], [], []
    
    if yerr is not None:
        theoNoiseNoBin = np.median(yerr)
    else:
        theoNoiseNoBin = np.nan
    
    cadence = np.median(np.diff(x))
    whiteNoiseStart = np.std(y)
    
    for oneBin in binNums:
        xBin,yBin,yBinErr = do_binning(x,y,nBin=oneBin)
        binSize = np.median(np.diff(xBin))
        nAvg = binSize/cadence
        
        stds.append(np.std(yBin))
        binSizes.append(binSize)
        theoNoise.append(theoNoiseNoBin / np.sqrt(nAvg))
        wNoise.append(whiteNoiseStart / np.sqrt(nAvg))

    ## do the unbinned Allan variance
    stds.append(whiteNoiseStart)
    binSizes.append(cadence)
    theoNoise.append(theoNoiseNoBin)
    wNoise.append(whiteNoiseStart)


    ## only Use finite values (ignore NaNs where binning isn't possible)
    usePts = np.isfinite(stds)

    fig, ax = plt.subplots(figsize=(5,4))
    if logPlot == True:
        ax.loglog(np.array(binSizes)[usePts],np.array(stds)[usePts],label='Measured')
    else:
        ax.semilogx(np.array(binSizes)[usePts],np.array(stds)[usePts],label='Measured')
    ax.plot(binSizes,theoNoise,label='Read + Photon Noise')
    ax.plot(binSizes,wNoise,label='White noise scaling')
    ax.set_xlabel('Bin Size ({})'.format(xUnit))
    ax.set_ylabel(r'$\sigma$ ({})'.format(yUnit))
    ax.set_ylim(yLim)
    ax.legend()
    ax.set_title("Allan Variance (Linear De-trend = {})".format(removeLinear))
    outName = 'all_var_{}_removelinear_{}.pdf'.format(customShortName,removeLinear)
    
    baseDir = get_baseDir()
    outPath = os.path.join(baseDir,'plots','allan_variance',outName)
    if showPlot == True:
        fig.show()
    else:
        fig.savefig(outPath,
                    bbox_inches='tight')
    
    
        plt.close(fig)

def exists_and_equal(dict1,key1,val1):
    """
    Simple test to see if 
    """
    if key1 in dict1:
        if dict1[key1] == val1:
            return True
        else:
            return False
    else:
        return False

def test_centroiding(useMultiprocessing=True):
    photObj = phot()
    photObj.fileL = photObj.fileL[0:30]; photObj.nImg = 30; photObj.param['doCentering'] = True
    photObj.get_allimg_cen(recenter=True,useMultiprocessing=useMultiprocessing)

def seeing_summary():
    """ Makes a summary of the seeing for a given night """
    cenList = glob.glob('centroids/*.fits')
    
    fileArr,medFWHMArr = [], []
    
    for oneFile in cenList:
        HDUList = fits.open(oneFile)
        if 'FWHM' in HDUList:
            fwhmData = np.abs(HDUList['FWHM'].data)
            low, med, high = np.nanpercentile(fwhmData,[0.32,0.5, 0.68])
            baseName = os.path.basename(oneFile)
            print("{}: {} {} {}".format(baseName,low,med,high))
            fileArr.append(baseName)
            medFWHMArr.append(med)
        HDUList.close()
    t = Table()
    t['File'] = fileArr
    t['FWHM'] = medFWHMArr
    return t


