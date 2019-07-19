import photutils
from ccdproc import CCDData, Combiner
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
import os
import warnings
from scipy.stats import binned_statistic
from astropy.table import Table
import multiprocessing
from multiprocessing import Pool
maxCPUs = multiprocessing.cpu_count() // 3


def run_one_phot(allInput):
    """
    Awkward workaround because multiprocessing doesn't work on object methods
    """
    photObj, ind = allInput
    return photObj.phot_for_one_file(ind)

def run_multiprocessing_phot(photObj,fileIndices):
    """
    Awkward workaround because multiprocessing doesn't work on object methods
    """
    allInput = []
    for oneInd in fileIndices:
        allInput.append([photObj,oneInd])
    p = Pool(maxCPUs)
    outputPhot = p.map(run_one_phot,allInput)
    return outputPhot

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
        """
        if directParam is None:
            self.paramFile = paramFile
            with open(paramFile) as pFile:
                self.param = yaml.load(pFile)
        else:
            self.paramFile = 'direct dictionary'
            self.param = directParam
            
        self.fileL = np.sort(glob.glob(self.param['procFiles']))
        self.nImg = len(self.fileL)
        xCoors, yCoors = [], []
        positions = self.param['refStarPos']
        self.nsrc = len(positions)
        
        defaultParams = {'srcGeometry': 'Circular', 'bkgSub': True, 'isCube': False, 'cubePlane': 0,
                        'doCentering': True, 'bkgGeometry': 'CircularAnnulus',
                        'boxFindSize': 18,'backStart': 9, 'backEnd': 12,
                        'scaleAperture': False, 'apScale': 2.5, 'apRange': [0.01,9999],
                        'nanTreatment': None, 'backOffset': [0.0,0.0],
                         'FITSextension': 0, 'HEADextension': 0,
                         'refPhotCentering': None,'isSlope': False,'readNoise': None,
                         'detectorGain': None}
        
        for oneKey in defaultParams.keys():
            if oneKey not in self.param:
                self.param[oneKey] = defaultParams[oneKey]
        
        
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
        self.srcNames[0] = 'src'
        
        self.set_up_apertures(positions)
        
        ## Set up file names for output
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        self.photFile = 'tser_data/phot/phot_'+self.dataFileDescrip+'.fits'
        self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'
        self.refCorPhotFile = 'tser_data/refcor_phot/refcor_'+self.dataFileDescrip+'.fits'
    
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
                        srcLabel=None):
        """ Show the star choices for photometry
        Parameters
        ------------------
        img: numpy 2D array
            (optional) An image to plot
        head: astropy FITS header
            (optional) hader for image
        custPos: numpy 2D array or list of tuple coordinates
            (optional) Custom positions
        showAps: bool
            (optional) Show apertures rather than circle stars
        srcLabel: str or None
            (optional) What should the source label be?
                        The default is "src"
        """
        fig, ax = plt.subplots()
        
        img, head = self.get_default_im(img=img,head=None)
        
        lowVmin = np.nanpercentile(img,1)
        highVmin = np.nanpercentile(img,99)

        imData = ax.imshow(img,cmap='viridis',vmin=lowVmin,vmax=highVmin,interpolation='nearest')
        ax.invert_yaxis()
        rad, txtOffset = 50, 20

        showApPos = self.get_default_cen(custPos=custPos)
        if showAps == True:
            self.srcApertures.plot(ax=ax)
            if self.param['bkgSub'] == True:
                self.bkgApertures.plot(ax=ax)
            outName = 'ap_labels_{}.pdf'.format(self.dataFileDescrip)
        else:
            ax.scatter(self.xCoors, self.yCoors, s=rad, facecolors='none', edgecolors='r')
            
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

    def showStamps(self,img=None,head=None,custPos=None,custFWHM=None):
        """Shows the fixed apertures on the image with postage stamps surrounding sources """ 
        
        ##  Calculate approximately square numbers of X & Y positions in the grid
        numGridY = int(np.floor(np.sqrt(self.nsrc)))
        numGridX = int(np.ceil(float(self.nsrc) / float(numGridY)))
        fig, axArr = plt.subplots(numGridY, numGridX)
        
        img, head = self.get_default_im(img=img,head=head)
        
        boxsize = self.param['boxFindSize']
        
        showApPos = self.get_default_cen(custPos=custPos)
        
        for ind, onePos in enumerate(showApPos):
            ax = axArr.ravel()[ind]
            
            yStamp = np.array(onePos[1] + np.array([-1,1]) * boxsize,dtype=np.int)
            xStamp = np.array(onePos[0] + np.array([-1,1]) * boxsize,dtype=np.int)
            
            stamp = img[yStamp[0]:yStamp[1],xStamp[0]:xStamp[1]]
            
            imData = ax.imshow(stamp,cmap='viridis',vmin=0,vmax=1.2e4,interpolation='nearest')
            ax.set_title(self.srcNames[ind])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            srcX, srcY = onePos[0] - xStamp[0],onePos[1] - yStamp[0]
            circ = plt.Circle((srcX,srcY),
                              self.param['apRadius'],edgecolor='red',facecolor='none')
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
        
            
        fig.show()
        fig.savefig('plots/photometry/postage_stamps/stamps_'+self.dataFileDescrip+'.pdf')
        
    def showCustSet(self,index=None,ptype='Stamps',defaultCen=False):
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
            self.showStamps(custPos=cen,img=img,head=head)
        elif ptype == 'Map':
            self.showStarChoices(custPos=cen,img=img,head=head)
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
    
    def save_centroids(self,cenArr,fwhmArr):
        """ Saves the image centroid data"""
        hdu = fits.PrimaryHDU(cenArr)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with centroids')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('dimension','dimension axis X=0,Y=1')
        hdu.header['AXIS2'] = ('src','source axis')
        hdu.header['AXIS3'] = ('image','image axis')
        hdu.header['BOXSZ'] = (self.param['boxFindSize'],'half-width of the box used for source centroids')
        hdu.header['REFCENS'] = (self.param['refPhotCentering'],'Reference Photometry file used to shift the centroids (or empty if none)')
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
    
    def get_allimg_cen(self,recenter=False):
        """ Get all image centroids
        If self.param['doCentering'] is False, it will just use the input aperture positions 
        """
        
        ndim=2 ## Number of dimensions in image (assuming 2D)
        
            
        if os.path.exists(self.centroidFile) and (recenter == False):
            HDUList = fits.open(self.centroidFile)
            cenArr, head = HDUList["CENTROIDS"].data, HDUList["CENTROIDS"].header
            if "FWHM" in HDUList:
                fwhmArr, headFWHM = HDUList["FWHM"].data, HDUList["FWHM"].header
                self.keepFWHM = True
            else:
                self.keepFWHM = False ## allow for legacy centroid files
            
            HDUList.close()
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
            #for ind, oneFile in enumerate(self.fileL):
            for ind, oneFile in enumerate(self.fileL):
                img, head = self.getImg(oneFile)
                allX, allY, allfwhmX, allfwhmY = self.get_allcen_img(img)
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
            self.backgOffsetArr = None
        
        if self.keepFWHM == True:
            self.fwhmArr = fwhmArr
            self.headFWHM = headFWHM
    
    def get_allcen_img(self,img,showStamp=False):
        """ Gets the centroids for all sources in one image """
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
        
        xcenSub,ycenSub = centroid_2dg(subimg)
        
        xcen = xcenSub + minX
        ycen = ycenSub + minY
        
        fwhmX, fwhmY = self.get_fwhm(subimg,xcenSub,ycenSub)
        
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
    
    def phot_for_one_file(self,ind):
        if np.mod(ind,15) == 0:
            print("On "+str(ind)+' of '+str(len(self.fileL)))
        oneImg = self.fileL[ind]
        img, head = self.getImg(oneImg)
        if 'DATE-OBS' in head:
            useDate = head['DATE-OBS']
        elif 'DATE' in head:
            warnings.warn('DATE-OBS not found in header. Using DATE instead')
            month1, day1, year1 = head['DATE'].split("/")
            useDate = "-".join([year1,month1,day1])                                                                              
        t0 = Time(useDate+'T'+head['TIME-OBS'])
        if 'timingMethod' in self.param:
            if self.param['timingMethod'] == 'JWSTint':
                t0 = t0 + (head['TFRAME'] + head['INTTIME']) * (head['ON_NINT']) * u.second
        
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
        
        if self.param['readNoise'] != None:
            readNoise = float(self.param['readNoise'])
        elif 'RDNOISE1' in head:
            readNoise = float(head['RDNOISE1'])
        else:
            readNoise = 1.0
            warnings.warn('Warning, no read noise specified')
        
        err = np.sqrt(np.abs(img) + readNoise**2) ## Should already be gain-corrected
        
        rawPhot = aperture_photometry(img,self.srcApertures,error=err)
        
        if self.param['bkgSub'] == True:
            self.bkgApertures.positions = self.cenArr[ind] + self.backgOffsetArr[ind]
            bkgPhot = aperture_photometry(img,self.bkgApertures,error=err)
            bkgVals = bkgPhot['aperture_sum'] / self.bkgApertures.area() * self.srcApertures.area()
            bkgValsErr = bkgPhot['aperture_sum_err'] / self.bkgApertures.area() * self.srcApertures.area()
        
            ## Background subtracted fluxes
            srcPhot = rawPhot['aperture_sum'] - bkgVals
        else:
            ## No background subtraction
            srcPhot = rawPhot['aperture_sum']
            bkgValsErr = 0.
            
        srcPhotErr = np.sqrt(rawPhot['aperture_sum_err']**2 + bkgValsErr**2)
        
        
        return [t0.jd,srcPhot,srcPhotErr]
    
    def return_self(self):
        return self
    
    def do_phot(self,useMultiprocessing=False):
        """ Does photometry using the centroids found in get_allimg_cen 
        """
        self.get_allimg_cen()
        
        photArr = np.zeros((self.nImg,self.nsrc))
        errArr = np.zeros_like(photArr)
        
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
            
        ## Save the photometry results
        hdu = fits.PrimaryHDU(photArr)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with photometry')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('src','source axis')
        hdu.header['AXIS2'] = ('image','image axis')
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
        
        hdu.header['BKGSUB'] = (self.param['bkgSub'], 'Do a background subtraction?')
        hdu.header['BKGSTART'] = (self.param['backStart'], 'Background Annulus start (px), if used')
        hdu.header['BKGEND'] = (self.param['backEnd'], 'Background Annulus end (px), if used')
        if 'backHeight' in self.param:
            hdu.header['BKHEIGHT'] = (self.param['backHeight'], 'Background Box Height (px)')
            hdu.header['BKWIDTH'] = (self.param['backWidth'], 'Background Box Width (px)')
        hdu.header['BKOFFSTX'] = (self.param['backOffset'][0], 'X Offset between background and source (px)')
        hdu.header['BKOFFSTY'] = (self.param['backOffset'][1], 'Y Offset between background and source (px)')
        
        hdu.header['BOXSZ'] = (self.param['boxFindSize'], 'half-width of the box used for centroiding')
        hdu.header['JDREF'] = (self.param['jdRef'], ' JD reference offset to subtract for plots')
        
        hdu.header['SCALEDAP'] = (self.param['scaleAperture'], 'Is the aperture scaled by the FWHM?')
        hdu.header['APSCALE'] = (self.param['apScale'], 'If scaling apertures, which scale factor?')
        hdu.header['ISCUBE'] = (self.param['isCube'], 'Is the image data 3D?')
        hdu.header['CUBPLANE'] = (self.param['cubePlane'], 'Which plane of the cube is used?')
        hdu.header['DOCEN'] = (self.param['doCentering'], 'Is each aperture centered individually?')
        
        hduFileNames = self.make_filename_hdu()
        
        hduTime = fits.ImageHDU(jdArr)
        hduTime.header['UNITS'] = ('days','JD time, UT')
        
        hduErr = fits.ImageHDU(data=errArr)
        hduErr.name = 'Phot Err'
        hduCen = fits.ImageHDU(data=self.cenArr,header=self.cenHead)
        
        hdu.name = 'Photometry'
        hduTime.name = 'Time'
        hduCen.name = 'Centroids'
        ## hduFileName.name = 'Filenames' # already named by make_filename_hdu
        
        HDUList = fits.HDUList([hdu,hduErr,hduTime,hduCen,hduFileNames])
        
        if self.keepFWHM == True:
            hduFWHM = fits.ImageHDU(self.fwhmArr,header=self.headFWHM)
            HDUList.append(hduFWHM)
        
        
        HDUList.writeto(self.photFile,overwrite=True)
        warnings.resetwarnings()
    
    def plot_phot(self,offset=0.,refCorrect=False,ax=None,fig=None,showLegend=True,
                  normReg=None,doBin=None,doNorm=True):
        """ Plots previously calculated photometry 
        Parameters
        ---------------------
        offset: float
            y displacement for overlaying time series
        refCorrect: bool
            Use reference star-corrected photometry?
        ax: matplotlib axis object
            If the axis was created separately, use the input axis object
        fig: matplotlib figure object
            If the figure was created separately, use the input axis object
        showLegend: bool
            Show a legend?
        normReg: list with two items or None
            Relative region over which to fit a baseline and re-normalize
            This only works on reference-corrected photometry for now
        doBin: float or None
            The bin size if showing binned data
            This only works on reference-corrected photometry for now
        doNorm: bool
            Normalize the individual time series?
        """
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        head = photHDU.header
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        timeHead = jdHDU.header
        
        jdRef = self.param['jdRef']
        
        if ax == None:
            fig, ax = plt.subplots()
        
        if refCorrect == True:
            yCorrected = self.refSeries(photArr)
            x = jdArr - jdRef
            if normReg == None:
                yShow = yCorrected
            else:
                fitp = (x < normReg[0]) | (x > normReg[1])
                polyBase = robust_poly(x[fitp],yCorrected[fitp],2,sigreject=2)
                yBase = np.polyval(polyBase,x)
                
                yShow = yCorrected / yBase
            ax.plot(x,yShow,label='data',marker='o',linestyle='',markersize=3.)
            
            if doBin is not None:
                minValue, maxValue = 0.98, 1.02 ## clip for cosmic rays
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
    
    def refSeries(self,photArr,reNorm=False,custSrc=None,sigRej=5.):
        """ Average together the reference stars
        Parameters
        -------------
        reNorm: bool
            Re-normalize all stars before averaging? If set all stars have equal weight
            Otherwise, the stars are summed together, which weights by flux
        custSrc: arr
            Custom sources to use in the averaging (to include/exclude specific sources)
        sigRej: int
            Sigma rejection threshold
        """
        combRef = []
        
        srcArray = np.arange(self.nsrc,dtype=np.int)
        
        if custSrc == None:
            refArrayTruth = (srcArray == 0)
        else:
            refArrayTruth = np.ones(self.nrc,dtype=np.bool)
            for ind, oneSrc in enumerate(custSrc):
                if oneSrc in srcArray:
                    refArrayTruth[ind] = False
        
        refMask = np.tile(refArrayTruth,(self.nImg,1))
        refPhot = np.ma.array(photArr,mask=refMask)
        
        ## Normalize all time series
        norm1D = np.nanmedian(photArr,axis=0)
        norm2D = np.tile(norm1D,(self.nImg,1))
        
        normPhot = refPhot / norm2D
        
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
        
        normPhot.mask = refMask | badP
        refPhot.mask = refMask | badP
        
        if reNorm == True:
            ## Weight all stars equally
            combRef = np.nanmean(normPhot,axis=1)
        else:
            ## Weight by the flux, but only for the valid points left
            weights = np.ma.array(norm2D,mask=normPhot.mask)
            ## Make sure weights sum to 1.0 for each time point (since some sources are missing)
            weightSums1D = np.nansum(weights,axis=1)
            weightSums2D = np.tile(weightSums1D,(self.nsrc,1)).transpose()
            weights = weights / weightSums2D
            combRef = np.nansum(normPhot * weights,axis=1)
        
        yCorrected = photArr[:,0] / combRef
        yCorrNorm = yCorrected / np.nanmedian(yCorrected)
        return yCorrNorm

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
        elif self.param['nanTreatment'] == '':
            raise NotImplementedError
        
        head = HDUList[headExtension].header
        if self.param['isSlope'] == True:
            if 'INTTIME' in head:
                intTime = head['INTTIME']
            else:
                warnings.warn("Couldn't find INTTIME in header. Trying EXPTIME")
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
        t['Y Corrected'] = self.refSeries(photArr)
        
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
        self.batchFile = batchFile
        with open(batchFile) as bFile:
            self.batchParam = yaml.load(bFile)
        
        ## Find keys that are lists. These are ones that are being run in batches
        ## However, a few keywords are already lists (like [x,y] coordinates))
        # and we are looking to see if those are lists of lists
        ## this dictionary specifies the depth of the list
        ## it could be a list of a list of list
        alreadyLists = {'refStarPos': 2,'backOffset': 1,'apRange': 1}
        self.paramLists = []
        self.counts = []
        for oneKey in self.batchParam.keys():
            if oneKey in alreadyLists:
                depth = alreadyLists[oneKey]
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
        
    def run_all(self,useMultiprocessing=False):
        for oneDict in self.paramDicts:
            thisPhot = phot(directParam=oneDict)
            print("Working on batch {} ".format(thisPhot.param['srcName'],
                                                thisPhot.dataFileDescrip))
            thisPhot.showStarChoices(showAps=True,srcLabel='0')
            thisPhot.do_phot(useMultiprocessing=useMultiprocessing)

    def plot_all(self):
        for oneDict in self.paramDicts:
            thisPhot = phot(directParam=oneDict)
            print("Working on batch {} ".format(thisPhot.param['srcName'],
                                                thisPhot.dataFileDescrip))
            thisPhot.plot_phot()
    
    def return_phot_obj(self,ind=0):
        """
        Return a photometry object so other methods and attributes can be explored
        """
        return phot(directParam=self.paramDicts[ind])
    

class prevPhot(phot):
    """ Loads in previous photometry from FITS data. Inherits functions from the phot class
    
    Parameters:
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
        
        keywordPath = os.path.join(os.path.dirname(__file__), 'parameters','phot_params',
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

def robust_poly(x,y,polyord,sigreject=3.0,iteration=3,useSpline=False,knots=None):
    finitep = np.isfinite(y) & np.isfinite(x)
    goodp = finitep ## Start with the finite points
    for iter in range(iteration):
        if np.sum(goodp) < polyord:
            warntext = "Less than "+str(polyord)+"points accepted, returning flat line"
            warnings.warn(warntext)
            coeff = np.zeros(polyord)
            coeff[0] = 1.0
        else:
            if useSpline == True:
                if knots is None:
                    spl = UnivariateSpline(x[goodp], y[goodp], k=polyord, s=sSpline)
                else:
                    spl = LSQUnivariateSpline(x[goodp], y[goodp], knots, k=polyord)
                ymod = spl(x)
            else:
                coeff = np.polyfit(x[goodp],y[goodp],polyord)
                yPoly = np.poly1d(coeff)
                ymod = yPoly(x)
            
            resid = np.abs(ymod - y)
            madev = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
            goodp = (np.abs(resid) < (sigreject * madev))
    
    if useSpline == True:
        return spl
    else:
        return coeff


