import photutils
from astropy.io import fits, ascii
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
import glob
from photutils import CircularAperture, CircularAnnulus
from photutils import RectangularAperture
from photutils import centroid_2dg, aperture_photometry
import photutils
import numpy as np
from astropy.time import Time
import astropy.units as u
import pdb
from copy import deepcopy
import yaml
import warnings
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from astropy.table import Table
import multiprocessing
from multiprocessing import Pool
import time
maxCPUs = multiprocessing.cpu_count() // 3
try:
    import bokeh.plotting
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.models import Range1d
    from bokeh.models import WheelZoomTool
except ImportError as err2:
    print("Could not import bokeh plotting. Interactive plotting may not work")


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
    if len(fileIndices) < maxCPUs:
        raise Exception("Fewer files to process than CPUs, this can confuse multiprocessing")
    
    p = Pool(maxCPUs)
    
    outputDat = p.map(run_one_phot_method,allInput)
    p.close()
    return outputDat

def read_yaml(filePath):
    with open(filePath) as yamlFile:
        yamlStructure = yaml.safe_load(yamlFile)
    return yamlStructure

class phot:
    def __init__(self,paramFile='parameters/phot_params/example_phot_parameters.yaml',
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
                        'nanTreatment': 'zero', 'backOffset': [0.0,0.0],
                        'srcName': 'WASP 62','srcNameShort': 'wasp62',
                         'refStarPos': [[50,50]],'procFiles': '*.fits',
                         'apRadius': 9,'FITSextension': 0,
                         'nightName': 'UT2020-01-20','srcName'
                         'FITSextension': 0, 'HEADextension': 0,
                         'refPhotCentering': None,'isSlope': False,
                         'itimeKeyword': 'INTTIME','readNoise': None,
                         'detectorGain': None,'cornerSubarray': False,
                         'subpixelMethod': 'exact','excludeList': None,
                         'dateFormat': 'Two Part','copyCentroidFile': None,
                         'bkgMethod': 'mean','diagnosticMode': False,
                         'bkgOrderX': 1, 'bkgOrderY': 1,'backsub_directions': ['Y','X'],
                         'saturationVal': None, 'satNPix': 5, 'nanReplaceValue': 0.0}
        
        
        for oneKey in defaultParams.keys():
            if oneKey not in self.param:
                self.param[oneKey] = defaultParams[oneKey]
        
        xCoors, yCoors = [], []
        positions = self.param['refStarPos']
        self.nsrc = len(positions)
        
        ## Set up file names for output
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        self.photFile = 'tser_data/phot/phot_'+self.dataFileDescrip+'.fits'
        self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'
        self.refCorPhotFile = 'tser_data/refcor_phot/refcor_'+self.dataFileDescrip+'.fits'
        
        # Get the file list
        self.fileL = self.get_fileList()
        self.nImg = len(self.fileL)
        
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
        self.srcNames[0] = 'src'
        
        self.set_up_apertures(positions)
        
        self.check_parameters()
        
    
    def get_parameters(self,paramFile,directParam=None):
        if directParam is None:
            self.paramFile = paramFile
            self.param = read_yaml(paramFile)
        else:
            self.paramFile = 'direct dictionary'
            self.param = directParam
    
    def get_fileList(self):
        origList = np.sort(glob.glob(self.param['procFiles']))
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
    
    def get_default_im(self,img=None,head=None):
        """ Get the default image for postage stamps or star identification maps"""
        ## Get the data
        if img is None:
            img, head = self.getImg(self.fileL[self.nImg // 2])
        
        return img, head
    
    def get_default_cen(self,custPos=None):
        """ Get the default centroids for postage stamps or star identification maps"""
        if custPos is None:
            showApPos = self.srcApertures.positions
        else:
            showApPos = custPos
        
        return showApPos
    
    def showStarChoices(self,img=None,head=None,custPos=None,showAps=False,
                        srcLabel=None,figSize=None):
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
        """
        fig, ax = plt.subplots(figsize=figSize)
        
        img, head = self.get_default_im(img=img,head=None)
        
        lowVmin = np.nanpercentile(img,1)
        highVmin = np.nanpercentile(img,99)

        imData = ax.imshow(img,cmap='viridis',vmin=lowVmin,vmax=highVmin,interpolation='nearest')
        ax.invert_yaxis()
        rad, txtOffset = 50, 20

        showApPos = self.get_default_cen(custPos=custPos)
        if showAps == True:
            apsShow = deepcopy(self.srcApertures)
            apsShow.positions = showApPos
            
            
            apsShow.plot(ax=ax)
            if self.param['bkgSub'] == True:
                backApsShow = deepcopy(self.bkgApertures)
                backApsShow.positions = showApPos
                backApsShow.positions[:,0] = backApsShow.positions[:,0] + self.param['backOffset'][0]
                backApsShow.positions[:,1] = backApsShow.positions[:,1] + self.param['backOffset'][1]
                backApsShow.plot(ax=ax)
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
            ax.text(onePos[0]+txtOffset,onePos[1]+txtOffset,name,color='white')
        
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(imData,label='Counts',cax=cax)
        fig.show()
        fig.savefig('plots/photometry/star_labels/{}'.format(outName),
                    bbox_inches='tight')
        plt.close(fig)

    def showStamps(self,img=None,head=None,custPos=None,custFWHM=None,
                   vmin=None,vmax=None,showPlot=False,boxsize=None):
        """Shows the fixed apertures on the image with postage stamps surrounding sources """ 
        
        ##  Calculate approximately square numbers of X & Y positions in the grid
        numGridY = int(np.floor(np.sqrt(self.nsrc)))
        numGridX = int(np.ceil(float(self.nsrc) / float(numGridY)))
        fig, axArr = plt.subplots(numGridY, numGridX)
        
        img, head = self.get_default_im(img=img,head=head)
        
        if boxsize == None:
            boxsize = self.param['boxFindSize']
        
        showApPos = self.get_default_cen(custPos=custPos)
        
        for ind, onePos in enumerate(showApPos):
            ax = axArr.ravel()[ind]
            
            yStamp = np.array(onePos[1] + np.array([-1,1]) * boxsize,dtype=np.int)
            xStamp = np.array(onePos[0] + np.array([-1,1]) * boxsize,dtype=np.int)
            
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
                              self.param['apRadius'],edgecolor='red',facecolor='none')
            ax.add_patch(circ)
            if self.param['bkgSub'] == True:
                for oneRad in [self.param['backStart'],self.param['backEnd']]:
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
        
        fig.savefig('plots/photometry/postage_stamps/stamps_'+self.dataFileDescrip+'.pdf')
        plt.close(fig)
        
    def showCustSet(self,index=None,ptype='Stamps',defaultCen=False,vmin=None,vmax=None):
        """ Show a custom stamp or star identification plot for a given image index 
        
        Parameters
        --------------
        index: int
            Index of the image/centroid to show
        ptype: str
            Plot type - 'Stamps' for postage stamps
                        'Map' for star identification map
        defaultCen: bool
            Use the default centroid? If true, it will use the guess centroids
        """
        self.get_allimg_cen()
        if index == None:
            index = self.nImg // 2
        
        img, head = self.getImg(self.fileL[index])
        
        if defaultCen == True:
            cen = self.srcApertures.positions
        else:
            cen = self.cenArr[index]
        
        if ptype == 'Stamps':
            self.showStamps(custPos=cen,img=img,head=head,vmin=vmin,vmax=vmax)
        elif ptype == 'Map':
            self.showStarChoices(custPos=cen,img=img,head=head,showAps=True)
        else:
            print('Unrecognized plot type')
            
    
    def make_filename_hdu(self):
        """
        Makes a Header data unit (binary FITS table) for filenames
        """
        fileLTable = Table()
        fileLTable['File Path'] = self.fileL
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
        if np.mod(ind,15) == 0:
            print("On {} of {}".format(ind,len(self.fileL)))
        img, head = self.getImg(self.fileL[ind])
        
        allX, allY = [], []
        allfwhmX, allfwhmY = [], []
        
        for ind, onePos in enumerate(self.srcApertures.positions):
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
        GaussModel = photutils.fit_2dgaussian(subimg)
        
        fwhmX = GaussModel.x_stddev.value * 2.35482005
        fwhmY = GaussModel.y_stddev.value * 2.35482005
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
    
    def phot_for_one_file(self,ind):
        if np.mod(ind,15) == 0:
            print("On "+str(ind)+' of '+str(len(self.fileL)))
        
        oneImg = self.fileL[ind]
        img, head = self.getImg(oneImg)
        
        t0 = self.get_date(head)
        
        self.srcApertures.positions = self.cenArr[ind]
        
        if 'scaleAperture' in self.param:
            if self.param['scaleAperture'] == True:
                medianFWHM = np.median(self.fwhmArr[ind])
                
                minFWHMallowed, maxFWHMallowed = self.param['apRange']
                bigRadius = 2. * self.param['backEnd']
                if medianFWHM < minFWHMallowed:
                    warnings.warn("FWHM found was smaller than apRange ({}) px. Using {} for Image {}".format(minFWHMallowed,minFWHMallowed,self.fileL[ind]))
                    medianFWHM = minFWHMallowed
                elif medianFWHM > maxFWHMallowed:
                    warnings.warn("FWHM found was larger than apRange ({}) px. Using {} for Image {}".format(maxFWHMallowed,maxFWHMallowed,self.fileL[ind]))
                    medianFWHM = maxFWHMallowed
                
                if self.param['bkgGeometry'] == 'CircularAnnulus':
                    self.srcApertures.r = medianFWHM * self.param['apScale']
                    self.bkgApertures.r_in = (self.srcApertures.r + 
                                              self.param['backStart'] - self.param['apRadius'])
                    self.bkgApertures.r_out = (self.bkgApertures.r_in +
                                               self.param['backEnd'] - self.param['backStart'])
                else:
                    warnings.warn('Background Aperture scaling not set up for non-annular geometry')
        
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
                bkgVals = bkgPhot['aperture_sum'] / self.bkgApertures.area() * self.srcApertures.area()
                bkgValsErr = bkgPhot['aperture_sum_err'] / self.bkgApertures.area() * self.srcApertures.area()
                
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
                
                bkgVals = np.array(bkgIntensity)  * self.srcApertures.area()
                bkgValsErr = np.array(bkgIntensityErr) * self.srcApertures.area()
            
                srcPhot = rawPhot['aperture_sum'] - bkgVals
            elif self.param['bkgMethod'] == 'colrow':
                srcPhot, bkgVals, bkgValsErr = self.poly_sub_phot(img,head,err,ind)
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
        bkgValsErr = bkgPhotTotal['aperture_sum_err'] / self.bkgApertures.area() * self.srcApertures.area()
        
        
        return np.array(srcPhot),np.array(bkgPhot),bkgValsErr
    
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
            for ind in fileCountArray:
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
                  excludeSrc=None,errBar=None):
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
                yNorm = yFlux / np.median(yFlux)
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
        
        fig.show()
        
        if refCorrect == True:
            fig.savefig('plots/photometry/tser_refcor/refcor_{}.pdf'.format(self.dataFileDescrip))
        else:
            fig.savefig('plots/photometry/tser_allstar/raw_tser_{}.pdf'.format(self.dataFileDescrip))
        
        HDUList.close()
        plt.close(fig)
    
    def print_phot_statistics(self,refCorrect=True,excludeSrc=None,shorten=False):
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        head = photHDU.header
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
            t['Mad (%)'] = np.round(mad * 100.,4)

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
            maskOut = np.zeros(self.nsrc,dtype=np.bool)
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
        
    def interactive_refSeries(self,excludeSrc=None):
        t, yCorr, yCorr_err = self.get_refSeries(excludeSrc=excludeSrc)
        outFile = "plots/photometry/interactive/refseries_{}.html".format(self.dataFileDescrip)
        
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
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
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
        fig.savefig('plots/photometry/tser_refcor/all_kic1255.pdf')
    else:
        fig.savefig('plots/photometry/tser_allstar/all_kic1255.pdf')


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
                   yUnit='ppm'):
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
    fig.savefig('plots/allan_variance/all_var_{}_removelinear_{}.pdf'.format(customShortName,removeLinear),
                bbox_inches='tight')
    
    
    plt.close(fig)

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

def robust_statistics(data,method='robust mean',nsig=10):
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    if method == 'median':
        oneStatistic = median_val
        err = mad / np.sqrt(np.sum(np.isfinite(data)))
    elif method == 'robust mean':
        goodp = np.abs(data - median_val) < (nsig * mad)
        oneStatistic = np.mean(data[goodp])
        err = mad / np.sqrt(np.sum(goodp))
    else:
        raise Exception("Unrecognized statistic {}".format(method))
    
    return oneStatistic, err

def robust_poly(x,y,polyord,sigreject=3.0,iteration=3,useSpline=False,knots=None,
                preScreen=False,plotEachStep=False):
    """
    Fit a function (with sigma rejection) to a curve
    
    Parameters
    -----------
    x: numpy array
        Independent variable
    y: numpy array
        Dependent variable
    polyord: int
        order of the fit (number of terms). polyord=1 is a linear fit,
        2 is a quadratic, etc.
    sigreject: float
        The 'sigma' rejection level in terms of median absolute deviations
    useSpline: bool
        Do a spline fit?
    knots: int or None
        How many knots to use if doing a spline fit
    preScreen: bool
        Pre-screen by removing outliers from the median (which might fail for large slopes)
    plotEachStep: bool
        Plot each step of the fitting?
    
    Example Usage:
    ----------
        import numpy as np
        from tshirt.pipeline import phot_pipeline
        import matplotlib.pyplot as plt
        
        x = np.arange(30)
        y = np.random.randn(30) + x
        y[2] = 80 ## an outlier
        polyfit = phot_pipeline.robust_poly(x,y,1)
        ymodel = np.polyval(polyfit,x)
        plt.plot(x,y,'o',label='input')
        plt.plot(x,ymodel,label='fit')
        plt.show()
        
    """
    finitep = np.isfinite(y) & np.isfinite(x)
    
    if preScreen == True:
        resid = np.abs(y - np.nanmedian(y))
        madev = np.nanmedian(resid)
        goodp = np.zeros_like(resid,dtype=np.bool)
        goodp[finitep] = (np.abs(resid[finitep]) < (sigreject * madev))
    else:
        goodp = finitep ## Start with the finite points
        
    for iter in range(iteration):
        if (useSpline == True) & (knots is not None):
            pointsThreshold = len(knots) + polyord
        else:
            pointsThreshold = polyord
        
        if np.sum(goodp) <= pointsThreshold:
            warntext = "Less than "+str(polyord)+"points accepted, returning flat line"
            warnings.warn(warntext)
            
            if useSpline == True:
                spl = UnivariateSpline([0,1,2],[0,0,0],k=1)
            else:
                coeff = np.zeros(polyord + 1)
                coeff[0] = 1.0
        else:
            if useSpline == True:
                
                if knots is None:
                    spl = UnivariateSpline(x[goodp], y[goodp], k=polyord, s=sSpline)
                else:
                    try:
                        spl = LSQUnivariateSpline(x[goodp], y[goodp], knots, k=polyord)
                    except ValueError as inst:
                        knownFailures = ((str(inst) == 'Interior knots t must satisfy Schoenberg-Whitney conditions') | 
                                         ("The input parameters have been rejected by fpchec." in str(inst)))
                        if knownFailures:
                            warnings.warn("Spline fitting failed because of Schoenberg-Whitney conditions. Trying to eliminate knots without sufficient data")
                            
                            if plotEachStep == True:
                                plt.plot(x[goodp],y[goodp],'o',label='data')
                                plt.plot(knots,np.ones_like(knots) * np.median(y[goodp]),'o',label='knots',markersize=10)
                            
                            keepKnots = np.zeros_like(knots,dtype=np.bool)
                            nKnots = len(knots)
                            for ind,oneKnot in enumerate(knots):
                                if ind == 0:
                                    if np.sum(x[goodp] < oneKnot) > 0:
                                        keepKnots[ind] = True
                                elif ind == nKnots - 1:
                                    if np.sum(x[goodp] > oneKnot) > 0:
                                        keepKnots[ind] = True
                                else:
                                    pointsTest = ((np.sum((x[goodp] > knots[ind-1]) & (x[goodp] < oneKnot)) > 0 ) &
                                                  (np.sum((x[goodp] > oneKnot) & (x[goodp] < knots[ind+1])) > 0 ))
                                    if pointsTest == True:
                                        keepKnots[ind] = True
                            if plotEachStep == True:
                                plt.plot(knots[keepKnots],np.ones_like(knots[keepKnots]) * np.median(y[goodp]),'o',label='knots to keep')
                                plt.show()
                            
                            knots = knots[keepKnots] 
                            spl = LSQUnivariateSpline(x[goodp], y[goodp], knots, k=polyord)
                            
                        else:
                            raise inst
                ymod = spl(x)
            else:
                coeff = np.polyfit(x[goodp],y[goodp],polyord)
                yPoly = np.poly1d(coeff)
                ymod = yPoly(x)
            
            resid = np.abs(ymod - y)
            madev = np.nanmedian(resid)
            if madev > 0:
                ## replacing the old line to avoid runtime errors
                ## goodp = (np.abs(resid) < (sigreject * madev))
                goodp = np.zeros_like(resid,dtype=np.bool)
                goodp[finitep] = (np.abs(resid[finitep]) < (sigreject * madev))
        
        if plotEachStep == True:
            plt.plot(x,y,'o')
            plt.plot(x[goodp],y[goodp],'o')
            plt.plot(x,ymod)
            plt.show()
    
    if useSpline == True:
        return spl
    else:
        return coeff


